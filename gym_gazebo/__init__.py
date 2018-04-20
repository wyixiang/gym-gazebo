from gym.envs.registration import register

register(
    id='GazeboTurtlebotLidar-v0',
    entry_point='gym_gazebo.envs.turtlebot:GazeboTurtlebotLidarEnv',
    # More arguments here
)
register(
    id='GazeboTurtlebotCameraEnv-v0',
    entry_point='gym_gazebo.envs.turtlebot:GazeboTurtlebotCameraEnv',
    # More arguments here
)