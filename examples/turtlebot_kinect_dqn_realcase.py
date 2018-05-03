
import time
import numpy as np
from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU

import subprocess

import rospy
import roslaunch
import os
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard

class DeepQ:
    """
    DQN abstraction.

    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s')

    """
    def __init__(self, inputs, outputs, learningRate):
        """
        Parameters:
            - inputs: input size
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.input_size = inputs
        self.output_size = outputs
        self.learningRate = learningRate
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

    def initNetworks(self, hiddenLayers):
        model = self.createModel(hiddenLayers, "relu", self.learningRate)
        self.model = model

    def createModel(self,hiddenLayers, activationType, learningRate):
        model = Sequential()
        if len(hiddenLayers) == 0:
            model.add(Dense(self.output_size, input_shape=(self.input_size,), kernel_initializer='lecun_uniform'))
            model.add(Activation("linear"))
        else :
            model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), kernel_initializer='lecun_uniform'))
            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))

            for index in range(1, len(hiddenLayers)):
                # print("adding layer "+str(index))
                layerSize = hiddenLayers[index]
                model.add(Dense(layerSize, kernel_initializer='lecun_uniform'))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
            model.add(Dense(self.output_size, kernel_initializer='lecun_uniform'))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()

        #plot_model(model, to_file='model-kinect-dqn.png')

        return model

    # predict Q values for all the actions
    def getQValues(self, state):
        predicted = self.model.predict(state.reshape(1,len(state)))
        return predicted[0]

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # select the action with the highest Q value
    def selectAction(self, qValues):
        action = self.getMaxIndex(qValues)
        return action

    def loadWeights(self, path):
        self.model.set_weights(load_model(path+'.h5').get_weights())


def discretize_observation(data,new_ranges):
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


class GoForward():
    def __init__(self):

        rospy.init_node('GoForward', anonymous=False)

        rospy.loginfo("To stop TurtleBot CTRL + C")

        #  when ctrl + c
        rospy.on_shutdown(self.shutdown)

        self.vel_pub = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=5)

        weights_path = './tmp/turtle_kinect_dqn1001400'

        learningRate = 0.00025
        network_inputs = 20
        network_outputs = 21
        network_layers = [300, 300]

        agent = DeepQ(network_inputs, network_outputs, learningRate)
        agent.initNetworks(network_layers)
        agent.loadWeights(weights_path)

        action_dim = network_outputs
        state_dim = network_inputs
        num=0

        while not rospy.is_shutdown():
            num+=1
            kinect_data = None
            while kinect_data is None:
                try:
                    kinect_data = rospy.wait_for_message('/kinect_scan', LaserScan, timeout=5)
                except:
                    pass
            observation = discretize_observation(kinect_data, state_dim)

            qValues = agent.getQValues(np.asarray(observation))

            action = agent.selectAction(qValues)

            max_ang_speed = 0.3
            action_mid = (action_dim - 1) / 2
            ang_vel = (action - action_mid) * max_ang_speed / action_mid  # from (-0.3 to + 0.3)

            vel_cmd = Twist()
            vel_cmd.linear.x = 0.2
            vel_cmd.angular.z = ang_vel
            print ("NUM" + "%3d" % num +" Action " + "%2d" % action + " linear " + "%.1f" % vel_cmd.linear.x + " angular " + "%3.1f" % vel_cmd.angular.z )

            self.vel_pub.publish(vel_cmd)

            time.sleep(0.4)

    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop TurtleBot")
        self.vel_pub.publish(Twist())
        rospy.sleep(1)


if __name__ == '__main__':
    try:
        GoForward()
    except:
        rospy.loginfo("GoForward node terminated.")
