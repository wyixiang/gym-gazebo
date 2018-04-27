import gym
import gym_gazebo
import time
from distutils.dir_util import copy_tree
import os
import json
import random
import numpy as np
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras import optimizers
from keras.layers import Conv2D, Flatten, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import MaxPooling2D
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
    def __init__(self, outputs, learningRate):
        """
        Parameters:
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.output_size = outputs
        self.learningRate = learningRate
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

    def initNetworks(self):
        model = self.createModel()
        self.model = model

    def createModel(self):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), strides=(2,2), input_shape=(img_rows,img_cols,img_channels)))
        model.add(Activation('relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(16, (3, 3), strides=(2,2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(self.output_size))
        #model.compile(RMSprop(lr=self.learningRate), 'MSE')
        optimizer = optimizers.RMSprop(lr=self.learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()

        #plot_model(model, to_file='model.png')

        return model

    # predict Q values for all the actions
    def getQValues(self, state):
        predicted = self.model.predict(state)
        return predicted[0]

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # select the action with the highest Q value
    def selectAction(self, observation):
        qValues = agent.getQValues(observation)
        action = self.getMaxIndex(qValues)
        return action

    def loadWeights(self, path):
        self.model.set_weights(load_model(path+'.h5').get_weights())
        print("success load")

if __name__ == '__main__':

    env = gym.make('GazeboTurtlebotCameraEnv-v0')
    outdir = '/tmp/gazebo_gym_experiments/'

    weights_path = './tmp/cbak/turtle_camera'

    img_rows, img_cols, img_channels = env.img_rows, env.img_cols, env.img_channels

    episode_count = 10000
    max_steps = 1000
    learningRate = 0.00025
    network_outputs = 21
    stepCounter = 0
    loadsim_seconds = 0

    agent = DeepQ(network_outputs, learningRate)
    agent.initNetworks()
    agent.loadWeights(weights_path)

    env.get_init(action_dim=network_outputs, max_step=max_steps)

    last10Scores = [0] * 10
    last10ScoresIndex = 0
    last10Filled = False
    highest_reward = 0
    start_time = time.time()

    #start iterating from 'current epoch'.
    for epoch in range(1, episode_count + 1, 1):
        observation = env.reset()
        state = observation
        cumulated_reward = 0

        for t in range(max_steps):
            env.get_step(t)

            action = agent.selectAction(state)

            newObservation, reward, done, info = env.step(action)
            newstate = newObservation

            cumulated_reward += reward
            #print(reward,cumulated_reward)
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            state = newstate

            stepCounter += 1

            if (t == max_steps-1):
                print ("reached the end")
                done = True

            if done:
                last10Scores[last10ScoresIndex] = cumulated_reward
                last10ScoresIndex += 1
                total_seconds = int(time.time() - start_time + loadsim_seconds)
                m, s = divmod(total_seconds, 60)
                h, m = divmod(m, 60)
                if last10ScoresIndex >= 10:
                    last10Filled = True
                    last10ScoresIndex = 0
                if not last10Filled:
                    print ("EP "+"%3d"%epoch +" -{:>4} steps".format(t+1)+" - CReward: "+"%5d"%cumulated_reward +"  Time: %d:%02d:%02d" % (h, m, s))
                else:
                    print ("EP " + str(epoch) +" -{:>4} steps".format(t+1) +" - last100 C_Rewards : " + str(int((sum(last10Scores) / len(last10Scores)))) + " - CReward: " + "%5d" % cumulated_reward + "  Eps=" + "  Time: %d:%02d:%02d" % (h, m, s))
                break

    env.close()

