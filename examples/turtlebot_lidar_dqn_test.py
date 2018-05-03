import gym
import gym_gazebo
import time
from distutils.dir_util import copy_tree
import os
import json
import random
import numpy as np
from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model
import memory
import liveplot

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
    def selectAction(self, state):
        qValues = agent.getQValues(state)
        action = self.getMaxIndex(qValues)
        return action

    def loadWeights(self, path):
        self.model.set_weights(load_model(path+'.h5').get_weights())


if __name__ == '__main__':

    env = gym.make('GazeboTurtlebotLidar-v0')

    weights_path = './tmp/dnn/turtle_lidar_dqn'

    episode_count = 10000
    max_steps = 1000
    learningRate = 0.00025
    network_inputs = 20
    network_outputs = 21
    network_layers = [300,300]
    stepCounter = 0
    loadsim_seconds = 0

    agent = DeepQ(network_inputs, network_outputs, learningRate)
    agent.initNetworks(network_layers)
    agent.loadWeights(weights_path)

    env.get_init(action_dim=network_outputs, state_dim=network_inputs, max_step=max_steps)

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    highest_reward = 0
    start_time = time.time()

    #start iterating from 'current epoch'.
    for epoch in range(1, episode_count + 1, 1):
        observation = env.reset()
        state = np.asarray(observation)
        cumulated_reward = 0

        for t in range(max_steps):
            env.get_step(t)

            action = agent.selectAction(state)

            newObservation, reward, done, info = env.step(action)
            newstate = np.asarray(newObservation)

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            state = newstate

            stepCounter += 1

            if (t == max_steps-1):
                print ("reached the end")
                done = True

            if done:
                last100Scores[last100ScoresIndex] = cumulated_reward
                last100ScoresIndex += 1
                total_seconds = int(time.time() - start_time + loadsim_seconds)
                m, s = divmod(total_seconds, 60)
                h, m = divmod(m, 60)
                if last100ScoresIndex >= 10:
                    last100Filled = True
                    last100ScoresIndex = 0
                if not last100Filled:
                    print ("EP "+"%3d"%epoch +" -{:>4} steps".format(t+1)+" - CReward: "+"%5d"%cumulated_reward +"  Time: %d:%02d:%02d" % (h, m, s))
                else:
                    print ("EP " + str(epoch) +" -{:>4} steps".format(t+1) +" - last100 C_Rewards : " + str(int((sum(last100Scores) / len(last100Scores)))) + " - CReward: " + "%5d" % cumulated_reward + "  Eps=" + "  Time: %d:%02d:%02d" % (h, m, s))
                break

    env.close()

