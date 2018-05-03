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

#### Real Turtlebot and Kinect
install turtlebot need 3 parts: rocon, kobuki, and turtlebot.
rocon don't have its apt package, so we install from source.
```bash
mkdir ~/rocon
cd ~/rocon
wstool init -j5 src https://raw.github.com/robotics-in-concert/rocon/release/indigo/rocon.rosinstall
source /opt/ros/indigo/setup.bash
rosdep install --from-paths src -i -y
catkin_make
```
```bash
sudo apt-get install ros-kinetic-turtlebot ros-kinetic-turtlebot-apps ros-kinetic-turtlebot-interactions ros-kinetic-kobuki-ftdi ros-kinetic-ar-track-alvar-msgs
```
build connect to turtlebot
```bash
roslaunch turtlebot_bringup minimal.launch
```
control from your keyboard
```bash
roslaunch turtlebot_teleop keyboard_teleop.launch
```
install kinectdriver
```bash
mkdir ~/kinectdriver 
cd ~/kinectdriver 
git clone https://github.com/avin2/SensorKinect 
cd SensorKinect/Bin/
tar xvjf SensorKinect093-Bin-Linux-x64-v5.1.2.1.tar.bz2
cd Sensor-Bin-Linux-x64-v5.1.2.1/
sudo ./install.sh
```
```bash
roslaunch openni_launch openni.launch
```
the driver above might not work, so we install another
```bash
sudo apt-get install libfreenect-dev ros-kinetic-freenect-launch
```
```bash
roslaunch freenect_launch freenect.launch
```
view image
```bash
rosrun image_view image_view image:=/camera/rgb/image_color
rosrun image_view image_view image:=/camera/depth/image
```
Kinect isn’t compatible with USB 3.0. Plug it into a USB 2.0 port. (Problems can also occur with USB 2.0 ports that are 3.0 compatible.).
Troubleshooting: I don’t have a USB 2.0 port or it’s still giving the above errors

We’re going to disable USB 3.0 in the BIOS.
```
Boot into BIOS by rebooting your computer and holding down the F2 key (it could be a different key on your computer).
Go to the advanced tab.
This part varies depending on your computer. If you have a USB option go there first.
If your BIOS lists xHCI, disable it.
If your BIOS lists USB Debugging, enable it.
Save and exit BIOS.
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
