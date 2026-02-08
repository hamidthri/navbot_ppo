#!/usr/bin/env python3
"""
Vision SAC Navigation Evaluation with Video Recording
Adapted from lidar evaluation script for vision+lidar fusion models
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
from sac_networks_vision import VisionActor
from vision_backbones import get_vision_backbone


# Test scenarios (same as lidar evaluation)
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


class VisionNavigationEvaluator:
    """Comprehensive navigation evaluation for vision models"""
    def __init__(self, model_path, device='cuda', record_video=True, output_dir=None, 
                 backbone='resnet18', fusion_type='film'):
        self.device = device
        self.record_video = record_video
        self.backbone = backbone
        self.fusion_type = fusion_type
        
        # Create results directory
        if output_dir:
            self.results_dir = output_dir
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.results_dir = f'evaluation_results_{timestamp}'
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'videos'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'trajectories'), exist_ok=True)
        
        # Initialize environment with vision
        rospy.init_node('vision_sac_evaluation', anonymous=True)
        self.env = Env(is_training=False, use_vision=True, reward_type='legacy')
        
        # Setup camera - environment already subscribes
        self.camera_available = True
        print("✓ Camera integrated in environment")
        
        # Load vision backbone
        print(f"\n{'='*60}")
        print("LOADING TRAINED MODEL")
        print(f"{'='*60}")
        
        vision_backbone = get_vision_backbone(
            backbone, 
            pretrained=True,
            output_dim=512
        )
        vision_backbone = vision_backbone.to(device)
        vision_backbone.eval()
        
        # Create policy network (actor only, no critic/replay buffer needed)
        self.policy = VisionActor(
            vision_backbone=vision_backbone,
            lidar_dim=16,
            action_dim=2,
            hidden_dim=256,
            fusion_type=fusion_type
        ).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Backbone: {backbone}, Fusion: {fusion_type}")
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
            rospy.sleep(0.3)
        except Exception as e:
            print(f"⚠ Failed to spawn target: {e}")
    
    def _delete_target(self):
        """Delete target model"""
        from gazebo_msgs.srv import DeleteModel
        
        try:
            if self.delete_model is None:
                rospy.wait_for_service('/gazebo/delete_model', timeout=5.0)
                self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            
            self.delete_model('target')
            self.target_spawned = False
            rospy.sleep(0.3)
        except Exception as e:
            print(f"⚠ Failed to delete target: {e}")
    
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
        # Spawn target at the goal position
        self._spawn_target(x, y)
        
        # Update environment's goal position
        self.env.goal_position.position.x = x
        self.env.goal_position.position.y = y
        
    def evaluate_scenario(self, scenario, max_steps=1000):
        """Evaluate one scenario"""
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario['name']} [{scenario['difficulty'].upper()}]")
        print(f"{'='*60}")
        print(f"  Start: {scenario['start']['region']} ({scenario['start']['x']}, {scenario['start']['y']})")
        print(f"  Goal:  {scenario['goal']['region']} ({scenario['goal']['x']}, {scenario['goal']['y']})")
        
        # Reset to specific start/goal
        self.set_robot_position(scenario['start']['x'], scenario['start']['y'])
        self.set_goal_position(scenario['goal']['x'], scenario['goal']['y'])
        
        # Let environment settle
        rospy.sleep(1.0)
        
        # Get initial state through environment step
        zero_action = np.array([0.0, 0.0])
        state_data, _, done, arrive = self.env.step(zero_action, zero_action)
        
        # Extract image and lidar from state
        if isinstance(state_data, dict):
            state = state_data['lidar']
            image = state_data['image']
        else:
            state = state_data
            image = None
        
        initial_distance = state[15]  # Current distance is at index 15
        print(f"  Initial distance: {initial_distance:.2f}m")
        
        # Start video recording
        if self.record_video:
            self.video_recorder.start_recording(scenario['name'])
        
        # Run episode
        trajectory = []
        step = 0
        total_reward = 0
        start_time = time.time()
        last_pos = None
        path_length = 0.0
        
        while step < max_steps and not done and not arrive:
            # Select action
            with torch.no_grad():
                if isinstance(state_data, dict) and 'image' in state_data:
                    # Vision + LiDAR
                    lidar_tensor = torch.FloatTensor(state[:16]).unsqueeze(0).to(self.device)
                    image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
                    action, _, _ = self.policy.sample(image_tensor, lidar_tensor)
                else:
                    # Fallback to lidar only (shouldn't happen but safe)
                    lidar_tensor = torch.FloatTensor(state[:16]).unsqueeze(0).to(self.device)
                    action, _, _ = self.policy.sample(None, lidar_tensor)
                
                action = action.cpu().numpy()[0]
                
                # Clip actions to valid ranges
                action[0] = np.clip(action[0], 0.0, 1.0)  # Linear velocity: [0, 1]
                action[1] = np.clip(action[1], -1.0, 1.0)  # Angular velocity: [-1, 1]
            
            # Execute action
            next_state_data, reward, done, arrive = self.env.step(action, action)
            
            # Extract image and lidar from state
            if isinstance(next_state_data, dict):
                next_state = next_state_data['lidar']
                image = next_state_data['image']
            else:
                next_state = next_state_data
                image = None
            
            # Track metrics
            current_distance = next_state[15]
            robot_x = self.env.position.x
            robot_y = self.env.position.y
            
            if last_pos is not None:
                path_length += np.sqrt((robot_x - last_pos[0])**2 + (robot_y - last_pos[1])**2)
            last_pos = (robot_x, robot_y)
            
            trajectory.append([robot_x, robot_y])
            
            # Record video frame
            if self.record_video and image is not None:
                elapsed_time = time.time() - start_time
                speed = path_length / elapsed_time if elapsed_time > 0 else 0.0
                metrics = {
                    'distance': current_distance,
                    'speed': speed,
                    'time': elapsed_time,
                    'status': 'navigating'
                }
                self.video_recorder.add_frame(image, metrics)
            
            total_reward += reward
            state = next_state
            state_data = next_state_data  # Keep dict for next iteration
            step += 1
            
            # Progress output
            if step % 50 == 0:
                elapsed = time.time() - start_time
                speed = path_length / elapsed if elapsed > 0 else 0.0
                print(f"  Step {step}: distance={current_distance:.2f}m, speed={speed:.2f}")
        
        # Stop video
        if self.record_video:
            self.video_recorder.stop_recording()
        
        # Save trajectory
        traj_file = os.path.join(self.results_dir, 'trajectories', 
                                f"{scenario['name'].replace(' ', '_')}.npz")
        np.savez(traj_file, positions=np.array(trajectory))
        
        # Determine result
        elapsed_time = time.time() - start_time
        if arrive:
            result_status = "SUCCESS"
            result_symbol = "✓"
        elif done:
            result_status = "COLLISION"
            result_symbol = "✗"
        else:
            result_status = "TIMEOUT"
            result_symbol = "⏱"
        
        distance_reduction = initial_distance - current_distance
        
        print(f"\n  RESULT:")
        print(f"  {result_symbol} {result_status}")
        print(f"  Steps: {step}")
        print(f"  Time: {elapsed_time:.1f}s")
        print(f"  Path length: {path_length:.2f}m")
        print(f"  Distance reduction: {distance_reduction:.2f}m")
        
        # Store results
        result = {
            'scenario': scenario['name'],
            'difficulty': scenario['difficulty'],
            'status': result_status,
            'success': arrive,
            'collision': done,
            'timeout': step >= max_steps,
            'steps': step,
            'time': elapsed_time,
            'path_length': path_length,
            'initial_distance': initial_distance,
            'final_distance': current_distance,
            'distance_reduction': distance_reduction,
            'total_reward': total_reward
        }
        self.all_results.append(result)
        
        return result
    
    def evaluate_all(self):
        """Evaluate all scenarios"""
        print(f"\n{'='*60}")
        print("STARTING COMPREHENSIVE EVALUATION")
        print(f"{'='*60}")
        print(f"Total scenarios: {len(TEST_SCENARIOS)}")
        print(f"Recording video: {'Yes' if self.record_video else 'No'}")
        
        for i, scenario in enumerate(TEST_SCENARIOS, 1):
            print(f"\n[{i}/{len(TEST_SCENARIOS)}]")
            self.evaluate_scenario(scenario)
            
            # Brief pause between scenarios
            rospy.sleep(2.0)
        
        # Generate report
        self._generate_report()
    
    def _generate_report(self):
        """Generate evaluation report"""
        print(f"\n{'='*60}")
        print("FINAL EVALUATION REPORT")
        print(f"{'='*60}\n")
        
        total = len(self.all_results)
        successes = sum(1 for r in self.all_results if r['success'])
        collisions = sum(1 for r in self.all_results if r['collision'])
        timeouts = sum(1 for r in self.all_results if r['timeout'])
        
        print(f"Total Scenarios:  {total}")
        print(f"Success:          {successes} ({100*successes/total:.1f}%)")
        print(f"Collision:        {collisions} ({100*collisions/total:.1f}%)")
        print(f"Timeout:          {timeouts} ({100*timeouts/total:.1f}%)")
        
        # By difficulty
        difficulties = {}
        for r in self.all_results:
            diff = r['difficulty']
            if diff not in difficulties:
                difficulties[diff] = {'total': 0, 'success': 0}
            difficulties[diff]['total'] += 1
            if r['success']:
                difficulties[diff]['success'] += 1
        
        print(f"\nPerformance by Difficulty:")
        for diff, stats in sorted(difficulties.items()):
            rate = 100 * stats['success'] / stats['total']
            print(f"  {diff.capitalize():8}: {stats['success']}/{stats['total']} ({rate:.1f}%)")
        
        # Detailed results table
        print(f"\nDetailed Results:")
        print(f"{'Scenario':<26} {'Result':<12} {'Time':<8} {'Steps':<8} {'Path':<8}")
        print("-" * 70)
        for r in self.all_results:
            status_symbol = "✓" if r['success'] else ("✗" if r['collision'] else "⏱")
            status = f"{status_symbol} {r['status']}"
            print(f"{r['scenario']:<26} {status:<12} {r['time']:<8.1f} {r['steps']:<8} {r['path_length']:<8.2f}")
        
        # Save report to file
        report_file = os.path.join(self.results_dir, 'evaluation_report.txt')
        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("VISION SAC EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total Scenarios:  {total}\n")
            f.write(f"Success:          {successes} ({100*successes/total:.1f}%)\n")
            f.write(f"Collision:        {collisions} ({100*collisions/total:.1f}%)\n")
            f.write(f"Timeout:          {timeouts} ({100*timeouts/total:.1f}%)\n\n")
            f.write("Performance by Difficulty:\n")
            for diff, stats in sorted(difficulties.items()):
                rate = 100 * stats['success'] / stats['total']
                f.write(f"  {diff.capitalize():8}: {stats['success']}/{stats['total']} ({rate:.1f}%)\n")
            f.write("\n")
            for r in self.all_results:
                f.write(f"{r}\n")
        
        print(f"\n✓ Report saved: {report_file}")
        if self.record_video:
            print(f"✓ Videos saved: {os.path.join(self.results_dir, 'videos')}")
        print(f"✓ Trajectories saved: {os.path.join(self.results_dir, 'trajectories')}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Vision SAC navigation')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--no-video', action='store_true', help='Disable video recording')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Vision backbone')
    parser.add_argument('--fusion', type=str, default='film', help='Fusion type')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = VisionNavigationEvaluator(
        model_path=args.model,
        device=args.device,
        record_video=not args.no_video,
        output_dir=args.output_dir,
        backbone=args.backbone,
        fusion_type=args.fusion
    )
    
    evaluator.evaluate_all()
