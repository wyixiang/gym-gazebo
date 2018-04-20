import rospy
import roslaunch
import numpy as np

from gym import spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan
from gym.utils import seeding

class GazeboTurtlebotLidarEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        #gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2TurtlebotLidar_v0.launch")
        #gazebo_env.GazeboEnv.__init__(self, "GazeboRoundTurtlebotLidar_v0.launch")
        gazebo_env.GazeboEnv.__init__(self, "GazeboMazeTurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
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

    def discretize_observation(self,data,new_ranges):
        discretized_ranges = []
        min_range = 0.2
        done = False
        mod = len(data.ranges)/new_ranges
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if data.ranges[i] == float ('Inf') or np.isinf(data.ranges[i]):
                    discretized_ranges.append(6)
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(int(data.ranges[i]))
            if (min_range > data.ranges[i] > 0):
                done = True

        return discretized_ranges,done

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

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state,done = self.discretize_observation(data,self.state_dim)

        if not done:
            # Straight reward = 5, Max angle reward = 0.5
            reward = round(15 * (max_ang_speed - abs(ang_vel) + 0.0335), 2)
        else:
            reward = -200

        if self.now_step == self.max_step - 1 :
            done = True

        return state, reward, done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
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
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state,done = self.discretize_observation(data,self.state_dim)

        return state
