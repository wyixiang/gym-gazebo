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
    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
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
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

    def initNetworks(self, hiddenLayers):
        model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.model = model

        #targetModel = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.targetModel = None

    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
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

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        print(file)
        os.unlink(file)

if __name__ == '__main__':

    env_o = gym.make('GazeboTurtlebotKinect-v0')
    outdir = '/tmp/gazebo_gym_experiments/'

    continue_execution = False

    weights_path = './tmp/turtle_kinect_dqn1001700'
    plotter = liveplot.LivePlot(outdir)

    episode_count = 10000
    max_steps = 1000
    updateTargetNetwork = 10000
    minibatch_size = 64
    learningRate = 0.00025
    discountFactor = 0.99
    memorySize = 10000
    learnStart = 64 # timesteps to observe before training
    network_inputs = 20
    network_outputs = 21
    network_layers = [300,300]
    explorationRate = 0
    current_epoch = 0
    stepCounter = 0
    loadsim_seconds = 0

    agent = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
    agent.initNetworks(network_layers)
    agent.loadWeights(weights_path)

    env_o.get_init(action_dim=network_outputs, state_dim=network_inputs, max_step=max_steps)
    env = gym.wrappers.Monitor(env_o, outdir,force=not continue_execution, resume=continue_execution)

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    highest_reward = 0
    start_time = time.time()

    #start iterating from 'current epoch'.
    for epoch in range(1, episode_count + 1, 1):
        observation = env.reset()
        cumulated_reward = 0

        # number of timesteps
        for t in range(max_steps):
            env_o.get_step(t)

            qValues = agent.getQValues(np.asarray(observation))

            action = agent.selectAction(qValues)

            newObservation, reward, done, info = env.step(action)

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            observation = newObservation

            stepCounter += 1

            if (t == max_steps-1):
                print ("reached the end")
                done = True

            env._flush(force=True)
            if done:
                last100Scores[last100ScoresIndex] = cumulated_reward
                last100ScoresIndex += 1
                total_seconds = int(time.time() - start_time + loadsim_seconds)
                m, s = divmod(total_seconds, 60)
                h, m = divmod(m, 60)
                if last100ScoresIndex >= 100:
                    last100Filled = True
                    last100ScoresIndex = 0
                if not last100Filled:
                    print ("EP "+"%3d"%epoch +" -{:>4} steps".format(t+1)+" - CReward: "+"%5d"%cumulated_reward +"  Eps="+"%3.2f"%explorationRate +"  Time: %d:%02d:%02d" % (h, m, s))
                else:
                    print ("EP " + str(epoch) +" -{:>4} steps".format(t+1) +" - last100 C_Rewards : " + str(int((sum(last100Scores) / len(last100Scores)))) + " - CReward: " + "%5d" % cumulated_reward + "  Eps=" + "%3.2f" % explorationRate + "  Time: %d:%02d:%02d" % (h, m, s))
                    if (epoch)%100==0:
                        env._flush()
                break


        #plotter.plot(env,average=5)

    env.close()

