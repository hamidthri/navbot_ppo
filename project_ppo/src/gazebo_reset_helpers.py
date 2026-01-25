# Gazebo Reset Helpers
import rospy
from gazebo_msgs.srv import SetModelState, GetModelState
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
import numpy as np
import time

def quaternion_from_yaw(yaw):
    qz = np.sin(yaw / 2.0)
    qw = np.cos(yaw / 2.0)
    return Quaternion(x=0.0, y=0.0, z=qz, w=qw)

def set_robot_pose(model_name, x, y, yaw, frame_id='world', z=0.07):
    """
    Set robot pose with physics pause to prevent falling through floor.
    
    Args:
        model_name: Name of the robot model in Gazebo
        x, y: Target position
        yaw: Target orientation (radians)
        frame_id: Reference frame (default: 'world')
        z: Height above ground (default: 0.07m - safe for Turtlebot3)
    """
    try:
        # Pause physics before teleporting
        rospy.wait_for_service('/gazebo/pause_physics', timeout=2.0)
        pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        pause()
        time.sleep(0.05)  # Brief pause to ensure physics stops
        
        # Set model state
        rospy.wait_for_service('/gazebo/set_model_state', timeout=2.0)
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        model_state = ModelState()
        model_state.model_name = model_name
        model_state.reference_frame = frame_id
        model_state.pose.position = Point(x=x, y=y, z=z)
        model_state.pose.orientation = quaternion_from_yaw(yaw)
        model_state.twist = Twist()  # Zero velocity
        resp = set_state(model_state)
        
        time.sleep(0.05)  # Brief pause to ensure state is set
        
        # Unpause physics
        rospy.wait_for_service('/gazebo/unpause_physics', timeout=2.0)
        unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpause()
        
        return resp.success
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logwarn(f'[gazebo_reset_helpers] Failed to set robot pose: {e}')
        return False

def move_goal_marker(marker_name, x, y, z=0.01, frame_id='world'):
    try:
        rospy.wait_for_service('/gazebo/set_model_state', timeout=2.0)
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        model_state = ModelState()
        model_state.model_name = marker_name
        model_state.reference_frame = frame_id
        model_state.pose.position = Point(x=x, y=y, z=z)
        model_state.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        model_state.twist = Twist()
        resp = set_state(model_state)
        return resp.success
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logwarn(f'[gazebo_reset_helpers] Failed to move goal marker: {e}')
        return False

def ensure_goal_marker_exists(marker_name='target'):
    try:
        rospy.wait_for_service('/gazebo/get_model_state', timeout=2.0)
        get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        resp = get_state(marker_name, 'world')
        return resp.success
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logwarn(f'[gazebo_reset_helpers] Goal marker {marker_name} not found: {e}')
        return False

def get_robot_pose(model_name, frame_id='world'):
    try:
        rospy.wait_for_service('/gazebo/get_model_state', timeout=2.0)
        get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        resp = get_state(model_name, frame_id)
        if resp.success:
            x = resp.pose.position.x
            y = resp.pose.position.y
            qz = resp.pose.orientation.z
            qw = resp.pose.orientation.w
            yaw = 2.0 * np.arctan2(qz, qw)
            return (x, y, yaw)
        else:
            return None
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logwarn(f'[gazebo_reset_helpers] Failed to get robot pose: {e}')
        return None
