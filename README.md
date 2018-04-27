## Installation


### Ubuntu 16.04
Basic requirements:
- ROS Kinetic
- Gazebo 8.1.1
- Python 2.7
- OpenCV
- OpenAI gym

#### install gym:
```bash
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```

#### install gazebo8:
```bash
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt-get install gazebo8
sudo apt-get install ros-kinetic-gazebo8-ros-pkgs ros-kinetic-gazebo8-ros-control
```

#### ROS Kinetic dependencies
```bash
pip install rospkg catkin_pkg

sudo apt-get install \
cmake gcc g++ qt4-qmake libqt4-dev \
libusb-dev libftdi-dev \
python3-defusedxml python3-vcstool python3-pyqt4 \
python-skimage \
ros-kinetic-octomap-msgs        \
ros-kinetic-joy                 \
ros-kinetic-geodesy             \
ros-kinetic-octomap-ros         \
ros-kinetic-control-toolbox     \
ros-kinetic-pluginlib	       \
ros-kinetic-trajectory-msgs     \
ros-kinetic-control-msgs	       \
ros-kinetic-std-srvs 	       \
ros-kinetic-nodelet	       \
ros-kinetic-urdf		       \
ros-kinetic-rviz		       \
ros-kinetic-kdl-conversions     \
ros-kinetic-eigen-conversions   \
ros-kinetic-tf2-sensor-msgs     \
ros-kinetic-pcl-ros \
ros-kinetic-navigation \
ros-kinetic-ar-track-alvar-msgs \
libspnav-dev libbluetooth-dev \
libcwiid-dev pyqt4-dev-tools libffi-dev \
ros-kinetic-depthimage-to-laserscan \
ros-kinetic-image-view 
```

#### Install Sophus
```bash
cd
git clone https://github.com/stonier/sophus -b release/0.9.1-kinetic
cd sophus
mkdir build
cd build
cmake ..
make
sudo make install
```

#### Gazebo gym

```bash
git clone https://github.com/wyixiang/gym-gazebo
cd gym-gazebo
sudo pip install -e .
```

#### Dependencies and libraries
```bash
# install Tensorflow
sudo pip install tensorflow
or
sudo pip install tensorflow-gpu

#install Keras
sudo pip install keras
```

Agent dependencies:
```bash
cd ~/gym-gazebo/gym_gazebo/envs/installation
bash setup_kinetic.bash	
```

remove cv2.so:
```bash
cd /opt/ros/kinetic/lib/python2.7/dist-packages/
sudo nautilus .
#do it using GUI
#remove or rename and check
pip install opencv-python
```

then:
```bash
vim ~/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/src/turtlebot_simulator/turtlebot_gazebo/launch/includes/kobuki.launch.xml
replace line6 to
<arg name="urdf_file" default="$(find xacro)/xacro.py '$(find turtlebot_description)/robots/$(arg base)_$(arg stacks)_$(arg 3d_sensor).urdf.xacro'"/>
```

Run the environment with a sample agent:
```bash
cd ~/gym-gazebo/gym_gazebo/envs/installation
bash turtlebot_setup.bash
cd ~/gym-gazebo/examples
export ROS_PORT_SIM=11311
python turtlebot_lidar_qlearn.py
```

to see the simulation:
```bash
gzclient
```

If you don't want to export everytime
```bash
echo "export ROS_PORT_SIM=11311" >> .bashrc
```

## Usage

### Build and install gym-gazebo

In the root directory of the repository:

```bash
sudo pip install -e .
```

### Running an environment

- Load the environment variables corresponding to the robot you want to launch. E.g. to load the Turtlebot:

```bash
cd ~/gym-gazebo/gym_gazebo/envs/installation
bash turtlebot_setup.bash
```

Note: all the setup scripts are available in `gym_gazebo/envs/installation`

- Run any of the examples available in `examples/`. E.g.:

```bash
cd examples
python turtlrbot_

### Display the simulation

To see what's going on in Gazebo during a simulation, simply run gazebo client:

```bash
gzclient
```

### Killing background processes

Sometimes, after ending or killing the simulation `gzserver` and `rosmaster` stay on the background, make sure you end them before starting new tests.

```bash
killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient
```
