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
    def __init__(self, outputs, memorySize, discountFactor, learningRate, learnStart):
        """
        Parameters:
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

    def initNetworks(self):
        model = self.createModel()
        self.model = model

        #targetModel = self.createModel()
        self.targetModel = None

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

    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print("layer ",i,": ",weights)
            i += 1

    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    # predict Q values for all the actions
    def getQValues(self, state):
        predicted = self.model.predict(state)
        return predicted[0]

    def getTargetQValues(self, state):
        predicted = self.targetModel.predict(state)
        return predicted[0]

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else :
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        return action

    def selectActionByProbability(self, qValues, bias):
        qValueSum = 0
        shiftBy = 0
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if (rand <= value):
                return i
            i += 1

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        # Do not learn until we've got self.learnStart samples
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            X_batch = np.empty((0,img_rows,img_cols,img_channels), dtype = np.float64)
            Y_batch = np.empty((0,self.output_size), dtype = np.float64)
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state)
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)
                else :
                    qValuesNewState = self.getQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

                X_batch = np.append(X_batch, state.copy(), axis=0)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_batch = np.append(X_batch, newState.copy(), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
            self.model.fit(X_batch, Y_batch, validation_split=0.2, batch_size = len(miniBatch), epochs=1, verbose = 0, callbacks=[TensorBoard(log_dir='./tmp/log')])

    def saveModel(self, path):
        self.model.save(path+'.h5')

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

    env_o = gym.make('GazeboTurtlebotCameraEnv-v0')
    outdir = '/tmp/gazebo_gym_experiments/'

    continue_execution = False

    weights_path = './tmp/turtle_camera'
    monitor_path = './tmp/turtle_camera'
    params_json = './tmp/turtle_camera.json'
    plotter = liveplot.LivePlot(outdir)

    img_rows, img_cols, img_channels = env_o.img_rows, env_o.img_cols, env_o.img_channels

    epsilon_discount = 0.999

    if not continue_execution:
        episode_count = 10000
        max_steps = 1000
        updateTargetNetwork = 10000
        minibatch_size = 64
        learningRate = 0.00025#1e-3
        discountFactor = 0.99
        memorySize = 10000
        learnStart = 500
        network_outputs = 21
        EXPLORE = 1500
        INITIAL_EPSILON = 1  # starting value of epsilon
        FINAL_EPSILON = 0.05  # final value of epsilon
        explorationRate = INITIAL_EPSILON
        current_epoch = 0
        stepCounter = 0
        loadsim_seconds = 0

        agent = DeepQ(network_outputs, memorySize, discountFactor, learningRate, learnStart)
        agent.initNetworks()

    else:
        #Load weights, monitor info and parameter info.
        with open(params_json) as outfile:
            d = json.load(outfile)
            episode_count = d.get('epochs')
            max_steps = d.get('steps')
            updateTargetNetwork = d.get('updateTargetNetwork')
            explorationRate = d.get('explorationRate')
            minibatch_size = d.get('minibatch_size')
            learnStart = d.get('learnStart')
            learningRate = d.get('learningRate')
            discountFactor = d.get('discountFactor')
            memorySize = d.get('memorySize')
            network_outputs = d.get('network_outputs')
            current_epoch = d.get('current_epoch')
            INITIAL_EPSILON = d.get('INITIAL_EPSILON')
            FINAL_EPSILON = d.get('FINAL_EPSILON')
            stepCounter = d.get('stepCounter')
            loadsim_seconds = d.get('loadsim_seconds')
            EXPLORE = d.get('EXPLORE')

        agent = DeepQ(network_outputs, memorySize, discountFactor, learningRate, learnStart)
        agent.initNetworks()
        agent.loadWeights(weights_path)

        clear_monitor_files(outdir)
        copy_tree(monitor_path, outdir)

    env_o.get_init(action_dim=network_outputs, max_step=max_steps)
    env = gym.wrappers.Monitor(env_o, outdir,force=not continue_execution, resume=continue_execution)

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    highest_reward = 0
    start_time = time.time()

    #start iterating from 'current epoch'.
    for epoch in range(current_epoch+1, episode_count + 1, 1):
        observation = env.reset()
        cumulated_reward = 0

        for t in range(max_steps):
            env_o.get_step(t)

            qValues = agent.getQValues(observation)

            action = agent.selectAction(qValues, explorationRate)

            newObservation, reward, done, info = env.step(action)

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            agent.addMemory(observation, action, reward, newObservation, done)
            observation = newObservation

            stepCounter += 1

            if stepCounter == learnStart:
                print("Starting learning")

            if stepCounter >= learnStart:
                agent.learnOnMiniBatch(minibatch_size, False)
                '''
                if stepCounter <= updateTargetNetwork:
                    deepQ.learnOnMiniBatch(minibatch_size, False)
                else :
                    deepQ.learnOnMiniBatch(minibatch_size, True)
                '''

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
                        agent.saveModel(weights_path)
                        env._flush()
                        copy_tree(outdir,monitor_path)
                        #save simulation parameters.
                        parameter_keys = ['epochs','steps','updateTargetNetwork','explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_outputs','current_epoch','stepCounter','EXPLORE','INITIAL_EPSILON','FINAL_EPSILON','loadsim_seconds']
                        parameter_values = [episode_count, max_steps, updateTargetNetwork, explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_outputs, epoch, stepCounter, EXPLORE, INITIAL_EPSILON, FINAL_EPSILON, total_seconds]
                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        with open(params_json, 'w') as outfile:
                            json.dump(parameter_dictionary, outfile)
                break

            '''
            if stepCounter % updateTargetNetwork == 0:
                deepQ.updateTargetNetwork()
                print ("updating target network")
            '''
        if epoch%10==0:
            plotter.plot(env,average=5)

        if explorationRate > FINAL_EPSILON and stepCounter > learnStart:
            explorationRate -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            #explorationRate *= epsilon_discount

    env.close()

