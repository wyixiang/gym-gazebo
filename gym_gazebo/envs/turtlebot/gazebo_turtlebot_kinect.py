import rospy
import roslaunch
import numpy as np

from gym import spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan
from gym.utils import seeding

class GazeboTurtlebotKinectEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2TurtlebotKinect_v0.launch")
        #gazebo_env.GazeboEnv.__init__(self, "GazeboRoundTurtlebotLidar_v0.launch")
        #gazebo_env.GazeboEnv.__init__(self, "GazeboMazeTurtlebotLidar_v1.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        '''
        self.action=3
        self.state_dim=5
        self.action_space = spaces.Discrete(3)
        self.action=21
        self.state_dim=20
        self.action_space = spaces.Discrete(21)
        '''
        self.reward_range = (-np.inf, np.inf)
        self.seed()

    def get_init(self, action_dim, state_dim, max_step, discrete_action=True):
        self.action = action_dim
        self.state_dim = state_dim
        self.max_step = max_step
        self.discrete_action=discrete_action
        if discrete_action:
            self.action_space = spaces.Discrete(self.action)
        else:
            self.action_space =(-1, 1)
        return

    def get_step(self, step):
        self.now_step=step
        return

    def ifdone(self, data):
        min_range = 0.21
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        return done

    def discretize_observation(self,data,new_ranges):
        discretized_ranges = []
        mod = len(data.ranges)/new_ranges
        for i in range(0, len(data.ranges), mod):
        #for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if data.ranges[i] == float ('Inf') or np.isinf(data.ranges[i]):
                    discretized_ranges.append(10)
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(int(data.ranges[i]))

        return discretized_ranges

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        max_ang_speed = 0.3
        if self.discrete_action:
            action_mid = (self.action - 1) / 2
            ang_vel = (action-action_mid)*max_ang_speed/action_mid #from (-0.3 to + 0.3)
        else:
            ang_vel = action * max_ang_speed

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.2
        vel_cmd.angular.z = ang_vel
        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        kinect_data = None
        while kinect_data is None:
            try:
                kinect_data = rospy.wait_for_message('/kinect_scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state = self.discretize_observation(kinect_data,self.state_dim)
        done = self.ifdone(data)

        laser_len = len(data.ranges)
        left_sum = sum(data.ranges[laser_len - (laser_len / 5):laser_len - (laser_len / 10)])  # 80-90
        right_sum = sum(data.ranges[(laser_len / 10):(laser_len / 5)])  # 10-20

        center_detour = abs(right_sum - left_sum) / 5

        if not done:
            # Straight reward = 5, Max angle reward = 0.5
            reward = round(15 * (max_ang_speed - abs(ang_vel) + 0.0335), 2)/(center_detour+1)
        else:
            reward = -200

        if self.now_step == self.max_step - 1 :
            done = True

        return state, reward, done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        #read laser data
        kinect_data = None
        while kinect_data is None:
            try:
                kinect_data = rospy.wait_for_message('/kinect_scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state = self.discretize_observation(kinect_data,self.state_dim)

        return state
