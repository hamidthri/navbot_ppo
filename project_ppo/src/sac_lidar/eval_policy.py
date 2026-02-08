#!/usr/bin/env python3
"""
SAC Navigation Evaluation with Video Recording
Compatible with lidar-only environment, captures video separately
"""
import os
import sys
import rospy
import numpy as np
import torch
import cv2
import time
from datetime import datetime
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge

from environment_small_house import Env
from sac import SAC
from sac_networks import GaussianPolicy


# Test scenarios
TEST_SCENARIOS = [
    {"name": "Cross-Map Easy", "start": {"x": 5.5, "y": -2.5, "region": "R1"}, "goal": {"x": -5.5, "y": 0.0, "region": "R5"}, "difficulty": "easy"},
    {"name": "Cross-Map Hard", "start": {"x": 7.0, "y": 2.5, "region": "R9"}, "goal": {"x": -7.0, "y": -4.5, "region": "R6"}, "difficulty": "hard"},
    {"name": "Long Corridor", "start": {"x": 3.5, "y": -4.0, "region": "R2"}, "goal": {"x": 7.5, "y": 2.5, "region": "R9"}, "difficulty": "medium"},
    {"name": "Narrow Passage", "start": {"x": 0.4, "y": -1.2, "region": "R11"}, "goal": {"x": -4.5, "y": 1.0, "region": "R7"}, "difficulty": "hard"},
    {"name": "U-Turn Challenge", "start": {"x": -7.5, "y": 1.5, "region": "R8"}, "goal": {"x": 3.2, "y": 2.3, "region": "R13"}, "difficulty": "hard"},
    {"name": "Central Area", "start": {"x": 0.0, "y": 1.5, "region": "R3"}, "goal": {"x": -1.5, "y": 5.0, "region": "R4"}, "difficulty": "easy"},
]


class VideoRecorder:
    """Records robot POV video with metrics overlay"""
    def __init__(self, save_dir, fps=10):
        self.save_dir = save_dir
        self.fps = fps
        self.writer = None
        self.frame_buffer = []
        
    def start_recording(self, scenario_name):
        """Start new video recording"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{scenario_name.replace(' ', '_')}_{timestamp}.mp4"
        filepath = os.path.join(self.save_dir, filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filepath, fourcc, self.fps, (640, 480))
        self.frame_buffer = []
        print(f"  🎥 Recording: {filename}")
        return filepath
    
    def add_frame(self, camera_image, metrics):
        """Add frame with metrics overlay"""
        if camera_image is None:
            return
            
        frame = camera_image.copy()
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Resize to 640x480
        frame = cv2.resize(frame, (640, 480))
        
        # Add semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Add metrics text
        y_offset = 25
        cv2.putText(frame, f"Distance: {metrics['distance']:.2f}m", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(frame, f"Speed: {metrics['speed']:.2f}m/s", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(frame, f"Time: {metrics['time']:.1f}s", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add status indicator
        status_color = (0, 255, 0) if metrics['status'] == 'navigating' else (0, 0, 255)
        cv2.circle(frame, (620, 20), 10, status_color, -1)
        
        if self.writer:
            self.writer.write(frame)
        self.frame_buffer.append(frame)
    
    def stop_recording(self):
        """Stop recording and save"""
        if self.writer:
            self.writer.release()
            self.writer = None
        print(f"  ✓ Video saved ({len(self.frame_buffer)} frames)")


class NavigationEvaluator:
    """Comprehensive navigation evaluation"""
    def __init__(self, model_path, device='cuda', record_video=True, output_dir=None):
        self.device = device
        self.record_video = record_video
        
        # Create results directory
        if output_dir:
            self.results_dir = output_dir
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.results_dir = f'evaluation_results_{timestamp}'
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'videos'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'trajectories'), exist_ok=True)
        
        # Initialize environment (no use_vision parameter!)
        rospy.init_node('sac_evaluation', anonymous=True)
        self.env = Env(is_training=False, reward_type='legacy')
        
        # Setup camera subscriber separately for video recording
        self.camera_image = None
        self.camera_available = False
        if self.record_video:
            try:
                self.bridge = CvBridge()
                self.sub_camera = rospy.Subscriber('/robot_camera/image_raw', Image, 
                                                   self.cameraCallback, queue_size=1)
                # Wait briefly to see if camera is available
                rospy.sleep(1.0)
                if self.camera_image is not None:
                    self.camera_available = True
                    print("✓ Camera available for video recording")
                else:
                    print("⚠ Camera not available - continuing without video")
                    self.record_video = False
            except Exception as e:
                print(f"⚠ Camera setup failed: {e}")
                self.record_video = False
        
        # Load trained model
        print(f"\n{'='*60}")
        print("LOADING TRAINED MODEL")
        print(f"{'='*60}")
        self.agent = SAC(
            state_dim=16,
            action_dim=2,
            device=device,
            hidden_dim=256,
            lr_actor=3e-4,
            lr_critic=3e-4
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        self.agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.agent.policy.eval()
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Results directory: {self.results_dir}")
        
        # Video recorder
        if self.record_video:
            self.video_recorder = VideoRecorder(
                os.path.join(self.results_dir, 'videos'),
                fps=10
            )
        
        # Statistics
        self.all_results = []
        
        # Setup Gazebo spawn/delete services
        self.spawn_model = None
        self.delete_model = None
        self.target_spawned = False
        
    def _spawn_target(self, x, y):
        """Spawn target at position"""
        from gazebo_msgs.srv import SpawnModel, DeleteModel
        import rospkg
        
        # Delete if already exists
        if self.target_spawned:
            self._delete_target()
        
        try:
            # Setup spawn service
            if self.spawn_model is None:
                rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=10.0)
                self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            
            # Load target model SDF
            rospack = rospkg.RosPack()
            model_path = rospack.get_path('turtlebot3_gazebo') + '/models/Target/model.sdf'
            with open(model_path, 'r') as f:
                model_xml = f.read()
            
            # Spawn target
            from geometry_msgs.msg import Pose
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.01
            pose.orientation.w = 1.0
            
            self.spawn_model('target', model_xml, '', pose, 'world')
            self.target_spawned = True
            rospy.sleep(0.3)  # Wait for spawn to complete
            
        except Exception as e:
            print(f"⚠ Failed to spawn target: {e}")
    
    def _delete_target(self):
        """Delete target model"""
        from gazebo_msgs.srv import DeleteModel
        
        try:
            if self.delete_model is None:
                rospy.wait_for_service('/gazebo/delete_model', timeout=10.0)
                self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            
            self.delete_model('target')
            self.target_spawned = False
            rospy.sleep(0.3)  # Wait for delete to complete
            
        except Exception as e:
            print(f"⚠ Failed to delete target: {e}")
    
    def cameraCallback(self, data):
        """Process camera image for video recording"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
            resized = cv2.resize(cv_image, (224, 224), interpolation=cv2.INTER_LINEAR)
            self.camera_image = resized.astype(np.uint8)
        except Exception as e:
            pass
        
    def set_robot_position(self, x, y):
        """Move robot to position"""
        from gazebo_msgs.srv import SetModelState
        from gazebo_msgs.msg import ModelState
        
        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        state_msg = ModelState()
        state_msg.model_name = 'turtlebot3_burger'
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.w = 1.0
        
        set_state(state_msg)
        time.sleep(0.5)
    
    def set_goal_position(self, x, y):
        """Spawn/move goal marker to position"""
        # Spawn target at the goal position (deletes if already exists)
        self._spawn_target(x, y)
        
        # Update environment's goal position
        self.env.goal_position.position.x = x
        self.env.goal_position.position.y = y
    
    def run_scenario(self, scenario):
        """Run single evaluation scenario"""
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario['name']} [{scenario['difficulty'].upper()}]")
        print(f"{'='*60}")
        print(f"  Start: {scenario['start']['region']} ({scenario['start']['x']:.1f}, {scenario['start']['y']:.1f})")
        print(f"  Goal:  {scenario['goal']['region']} ({scenario['goal']['x']:.1f}, {scenario['goal']['y']:.1f})")
        
        # Setup
        self.set_robot_position(scenario['start']['x'], scenario['start']['y'])
        self.set_goal_position(scenario['goal']['x'], scenario['goal']['y'])
        
        # Get initial state
        data = rospy.wait_for_message('scan', LaserScan, timeout=5)
        scan_range, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.env.getState(data)
        
        initial_distance = rel_dis
        print(f"  Initial distance: {initial_distance:.2f}m")
        
        # Start video recording
        video_path = None
        if self.record_video and self.camera_available:
            video_path = self.video_recorder.start_recording(scenario['name'])
        
        # Navigation loop
        trajectory = []
        state = [i / 3.5 for i in scan_range] + [0, 0, rel_dis/20, yaw/360, rel_theta/360, diff_angle/180]
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        max_steps = 500
        step_count = 0
        success = False
        collision = False
        start_time = time.time()
        
        past_action = np.array([0.0, 0.0])
        
        while step_count < max_steps:
            # Select action with small exploration noise (like training)
            with torch.no_grad():
                action, _, _ = self.agent.policy.sample(state)
                action = action.cpu().numpy()[0]
                
                # Add small Gaussian noise for exploration (helps reach goal)
                action[0] += np.random.normal(0, 0.1)  # Linear velocity noise
                action[1] += np.random.normal(0, 0.05)   # Angular velocity noise
                
                action[0] = np.clip(action[0], 0.0, 1.0)
                action[1] = np.clip(action[1], -1.0, 1.0)
            
            # Execute action
            next_state, reward, done, arrive = self.env.step(action, past_action)
            
            # Get current metrics
            current_time = time.time() - start_time
            current_distance = np.sqrt((self.env.goal_position.position.x - self.env.position.x)**2 +
                                      (self.env.goal_position.position.y - self.env.position.y)**2)
            speed = action[0]
            
            # Record trajectory
            trajectory.append({
                'x': self.env.position.x,
                'y': self.env.position.y,
                'time': current_time,
                'distance': current_distance
            })
            
            # Record video frame
            if self.record_video and self.camera_available and self.camera_image is not None:
                metrics = {
                    'distance': current_distance,
                    'speed': speed * 0.22,  # Convert to m/s
                    'time': current_time,
                    'status': 'navigating'
                }
                self.video_recorder.add_frame(self.camera_image, metrics)
            
            # Check termination
            if arrive:
                success = True
                break
            if done:
                collision = True
                break
            
            # Update state
            past_action = action
            state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            step_count += 1
            
            # Print progress every 50 steps
            if step_count % 50 == 0:
                print(f"  Step {step_count}: distance={current_distance:.2f}m, speed={speed:.2f}")
        
        # Stop recording
        if self.record_video and self.camera_available:
            self.video_recorder.stop_recording()
        
        # Calculate results
        elapsed_time = time.time() - start_time
        final_distance = current_distance
        path_length = sum(np.sqrt((trajectory[i+1]['x'] - trajectory[i]['x'])**2 + 
                                  (trajectory[i+1]['y'] - trajectory[i]['y'])**2)
                         for i in range(len(trajectory)-1))
        
        # Save trajectory
        traj_file = os.path.join(self.results_dir, 'trajectories', 
                                f"{scenario['name'].replace(' ', '_')}.txt")
        with open(traj_file, 'w') as f:
            f.write("time,x,y,distance\n")
            for t in trajectory:
                f.write(f"{t['time']:.2f},{t['x']:.3f},{t['y']:.3f},{t['distance']:.3f}\n")
        
        # Results
        result = {
            'scenario': scenario['name'],
            'difficulty': scenario['difficulty'],
            'success': success,
            'collision': collision,
            'timeout': not success and not collision,
            'initial_distance': initial_distance,
            'final_distance': final_distance,
            'distance_reduction': initial_distance - final_distance,
            'path_length': path_length,
            'steps': step_count,
            'time': elapsed_time,
            'video_path': video_path
        }
        
        # Print summary
        print(f"\n  RESULT:")
        if success:
            print(f"  ✓ SUCCESS")
        elif collision:
            print(f"  ✗ COLLISION")
        else:
            print(f"  ⊙ TIMEOUT")
        print(f"  Steps: {step_count}")
        print(f"  Time: {elapsed_time:.1f}s")
        print(f"  Path length: {path_length:.2f}m")
        print(f"  Distance reduction: {result['distance_reduction']:.2f}m")
        
        return result
    
    def evaluate_all(self):
        """Run all test scenarios"""
        print(f"\n{'='*60}")
        print("STARTING COMPREHENSIVE EVALUATION")
        print(f"{'='*60}")
        print(f"Total scenarios: {len(TEST_SCENARIOS)}")
        print(f"Recording video: {'Yes' if self.record_video and self.camera_available else 'No'}")
        
        for i, scenario in enumerate(TEST_SCENARIOS, 1):
            print(f"\n[{i}/{len(TEST_SCENARIOS)}]")
            result = self.run_scenario(scenario)
            self.all_results.append(result)
            time.sleep(2)  # Brief pause between scenarios
        
        # Generate final report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        print(f"\n{'='*60}")
        print("FINAL EVALUATION REPORT")
        print(f"{'='*60}\n")
        
        # Overall statistics
        total = len(self.all_results)
        successes = sum(1 for r in self.all_results if r['success'])
        collisions = sum(1 for r in self.all_results if r['collision'])
        timeouts = sum(1 for r in self.all_results if r['timeout'])
        
        success_rate = 100 * successes / total if total > 0 else 0
        
        print(f"Total Scenarios:  {total}")
        print(f"Success:          {successes} ({success_rate:.1f}%)")
        print(f"Collision:        {collisions} ({100*collisions/total:.1f}%)")
        print(f"Timeout:          {timeouts} ({100*timeouts/total:.1f}%)")
        print()
        
        # Per-difficulty analysis
        difficulties = {}
        for r in self.all_results:
            diff = r['difficulty']
            if diff not in difficulties:
                difficulties[diff] = {'total': 0, 'success': 0}
            difficulties[diff]['total'] += 1
            if r['success']:
                difficulties[diff]['success'] += 1
        
        print("Performance by Difficulty:")
        for diff in ['easy', 'medium', 'hard']:
            if diff in difficulties:
                d = difficulties[diff]
                rate = 100 * d['success'] / d['total']
                print(f"  {diff.capitalize():8s}: {d['success']}/{d['total']} ({rate:.1f}%)")
        print()
        
        # Detailed results
        print("Detailed Results:")
        print(f"{'Scenario':<25s} {'Result':<12s} {'Time':<8s} {'Steps':<7s} {'Path':<8s}")
        print("-" * 70)
        for r in self.all_results:
            status = "✓ Success" if r['success'] else ("✗ Collision" if r['collision'] else "⊙ Timeout")
            print(f"{r['scenario']:<25s} {status:<12s} {r['time']:<8.1f} {r['steps']:<7d} {r['path_length']:<8.2f}")
        
        # Save report to file
        report_file = os.path.join(self.results_dir, 'evaluation_report.txt')
        with open(report_file, 'w') as f:
            f.write("SAC NAVIGATION EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Scenarios: {total}\n")
            f.write(f"Success Rate: {success_rate:.1f}%\n\n")
            
            f.write("Scenario Details:\n")
            for r in self.all_results:
                f.write(f"\n{r['scenario']}:\n")
                f.write(f"  Result: {'Success' if r['success'] else 'Failed'}\n")
                f.write(f"  Time: {r['time']:.1f}s\n")
                f.write(f"  Steps: {r['steps']}\n")
                f.write(f"  Path length: {r['path_length']:.2f}m\n")
                if r['video_path']:
                    f.write(f"  Video: {os.path.basename(r['video_path'])}\n")
        
        print(f"\n✓ Report saved: {report_file}")
        if self.record_video and self.camera_available:
            print(f"✓ Videos saved: {os.path.join(self.results_dir, 'videos')}")
        print(f"✓ Trajectories saved: {os.path.join(self.results_dir, 'trajectories')}")
        print(f"\n{'='*60}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate SAC navigation')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--no-video', action='store_true', help='Disable video recording')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = NavigationEvaluator(
        model_path=args.model,
        device=args.device,
        record_video=not args.no_video,
        output_dir=args.output_dir
    )
    
    evaluator.evaluate_all()