import numpy as np
import gym
import gym_gazebo
import tensorflow as tf
import os
from keras.models import Sequential, Model, load_model
from keras import initializers
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Input, merge
import keras.backend as K
from distutils.dir_util import copy_tree
import json

from keras.utils import plot_model


import time

import memory
import liveplot


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        S = Input(shape=[state_size])
        h0 = Dense(100, init='he_uniform',activation='relu')(S)
        h1 = Dense(100, init='he_uniform',activation='relu')(h0)
        V = Dense(action_dim, activation='tanh', kernel_initializer= initializers.RandomNormal(
            mean=0.0, stddev=0.9, seed=None))(h1)
        model = Model(input=S,output=V)
        return model, model.trainable_weights, S


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        self.HIDDEN1_UNITS = 100
        self.HIDDEN2_UNITS = 100

        K.set_session(sess)

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim], name='action2')
        w = Dense(self.HIDDEN1_UNITS, init='he_uniform', activation='relu')(S)
        h = merge([w, A], mode='concat')
        h3 = Dense(self.HIDDEN2_UNITS, init='he_uniform', activation='relu')(h)
        V = Dense(action_dim, activation='linear')(h3)
        model = Model(input=[S, A], output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S


class DDPG:
    def __init__(self, state_dim, action_dim, memorySize, discountFactor, learnStart, minibatch_size,TAU,LRA,LRC):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = memory.Memory(memorySize)
        self.GAMMA = discountFactor
        self.learnStart = learnStart
        self.minibatch_size = minibatch_size

        self.TAU = TAU  # Target Network HyperParameters
        self.LRA = LRA  # Learning rate for Actor
        self.LRC = LRC  # Lerning rate for Critic

    def initNetworks(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

        self.actor = ActorNetwork(sess, state_dim, action_dim, self.minibatch_size, self.TAU, self.LRA)
        self.critic = CriticNetwork(sess, state_dim, action_dim, self.minibatch_size, self.TAU, self.LRC)

        self.writer = tf.summary.FileWriter('./tmp/log-ddpg', sess.graph)

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnMiniBatch(self, miniBatchSize):
        if self.memory.getCurrentSize() > self.learnStart:
            batch = self.memory.getMiniBatch(miniBatchSize)
            states = np.asarray([e['state'] for e in batch])
            actions = np.asarray([e['action'] for e in batch])
            rewards = np.asarray([e['reward'] for e in batch])
            new_states = np.asarray([e['newState'] for e in batch])
            dones = np.asarray([e['isFinal'] for e in batch])
            y_t = np.asarray([e['action'] for e in batch])

            target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + self.GAMMA * target_q_values[k]

            actions=actions.reshape(minibatch_size)
            y_t = y_t.reshape(minibatch_size)
            #??????????????????bug
            loss = self.critic.model.train_on_batch([states, actions], y_t)
            a_for_grad = self.actor.model.predict(states)
            grads = self.critic.gradients(states, a_for_grad)
            self.actor.train(states, grads)
            self.actor.target_train()
            self.critic.target_train()

            summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss), ])
            self.writer.add_summary(summary)

    def saveModel(self, path):
        self.actor.model.save(path+'_actor.h5')
        self.actor.target_model.save(path + '_actor_target.h5')
        self.critic.model.save(path+'_critic.h5')
        self.critic.target_model.save(path + '_critic_target.h5')

    def loadWeights(self, path):
        self.actor.model.set_weights(load_model(path+'_actor.h5').get_weights())
        self.actor.target_model.set_weights(load_model(path + '_actor_target.h5').get_weights())
        self.critic.model.set_weights(load_model(path+'_critic.h5').get_weights())
        self.critic.target_model.set_weights(load_model(path + '_critic_target.h5').get_weights())


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

    env_o = gym.make('GazeboTurtlebotLidar-v0')
    outdir = '/tmp/gazebo_gym_experiments/'

    continue_execution = False

    weights_path = './tmp/turtle_lidar_ddpg'
    monitor_path = './tmp/turtle_lidar_ddpg'
    params_json = './tmp/turtle_lidar_ddpg.json'
    plotter = liveplot.LivePlot(outdir)

    EXPLORE = 1000

    if not continue_execution:
        episode_count = 10000
        max_steps = 1000
        minibatch_size = 64
        discountFactor = 0.99
        memorySize = 10000
        learnStart = 64 # timesteps to observe before training
        action_dim = 1
        state_dim = 20
        INITIAL_EPSILON = 1  # starting value of epsilon
        FINAL_EPSILON = 0.05  # final value of epsilon
        explorationRate = INITIAL_EPSILON
        current_epoch = 0
        stepCounter = 0
        loadsim_seconds = 0
        TAU = 0.001  # Target Network HyperParameters
        LRA = 0.0001  # Learning rate for Actor
        LRC = 0.001  # Lerning rate for Critic

        agent = DDPG(state_dim, action_dim, memorySize, discountFactor, learnStart, minibatch_size,TAU,LRA,LRC)
        agent.initNetworks()

    else:
        #Load weights, monitor info and parameter info.
        with open(params_json) as outfile:
            d = json.load(outfile)
            episode_count = d.get('epochs')
            max_steps = d.get('steps')
            explorationRate = d.get('explorationRate')
            minibatch_size = d.get('minibatch_size')
            learnStart = d.get('learnStart')
            action_dim = d.get('action_dim')
            state_dim = d.get('state_dim')
            discountFactor = d.get('discountFactor')
            memorySize = d.get('memorySize')
            current_epoch = d.get('current_epoch')
            INITIAL_EPSILON = d.get('INITIAL_EPSILON')
            FINAL_EPSILON = d.get('FINAL_EPSILON')
            stepCounter = d.get('stepCounter')
            loadsim_seconds = d.get('loadsim_seconds')
            TAU = d.get('TAU')  # Target Network HyperParameters
            LRA = d.get('LRA')  # Learning rate for Actor
            LRC = d.get('LRC')  # Lerning rate for Critic

        agent = DDPG(state_dim, action_dim, memorySize, discountFactor, learnStart, minibatch_size,TAU,LRA,LRC)
        agent.initNetworks()
        agent.loadWeights(weights_path)

        clear_monitor_files(outdir)
        copy_tree(monitor_path, outdir)

    env_o.get_init(action_dim=action_dim, state_dim=state_dim, max_step=max_steps,discrete_action=False)
    env = gym.wrappers.Monitor(env_o, outdir, force=not continue_execution, resume=continue_execution)

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    highest_reward = 0
    start_time = time.time()

    #start iterating from 'current epoch'.
    for epoch in range(current_epoch+1, episode_count + 1, 1):
        observation = env.reset()
        state = np.array(observation)
        cumulated_reward = 0

        for t in range(max_steps):
            env_o.get_step(t)
            
            if np.random.random() > explorationRate:
                a_type = "Exploit"
                action = agent.actor.model.predict(state.reshape(1, len(state)))
            else:
                a_type = "Explore"
                action = np.random.uniform(-1, 1)

            newObservation, reward, done, info = env.step(action)
            newstate = np.array(newObservation)

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            agent.addMemory(state, action, reward, newstate, done)

            state = newstate

            stepCounter += 1

            if stepCounter == learnStart:
                print("Starting learning")

            if stepCounter >= learnStart:
                agent.learnOnMiniBatch(minibatch_size)

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
                    print ("EP "+"%3d"%epoch +" -{:>4} steps".format(t+1)+" - CReward: "+"%5d"%cumulated_reward +"  Eps="+"%3.2f"%explorationRate +"  TotalStep: " + "%4d"%stepCounter+"  Time: %d:%02d:%02d" % (h, m, s))
                else:
                    print ("EP " + str(epoch) +" -{:>4} steps".format(t+1) +" - last100 C_Rewards : " + str(int((sum(last100Scores) / len(last100Scores)))) + " - CReward: " + "%5d" % cumulated_reward + "  Eps=" + "%3.2f" % explorationRate +"  TotalStep: " + str(stepCounter)+ "  Time: %d:%02d:%02d" % (h, m, s))
                    if (epoch)%100==0:
                        agent.saveModel(weights_path)
                        env._flush()
                        copy_tree(outdir,monitor_path)
                        #save simulation parameters.
                        parameter_keys = ['epochs','steps','explorationRate','minibatch_size','learnStart','discountFactor','memorySize','current_epoch','stepCounter','INITIAL_EPSILON','FINAL_EPSILON','loadsim_seconds','action_dim','state_dim','TAU','LRC','LRA']
                        parameter_values = [episode_count, max_steps, explorationRate, minibatch_size, learnStart, discountFactor, memorySize, epoch, stepCounter, INITIAL_EPSILON, FINAL_EPSILON, total_seconds, action_dim, state_dim, TAU, LRC, LRA]
                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        with open(params_json, 'w') as outfile:
                            json.dump(parameter_dictionary, outfile)
                break

        if epoch%10==0:
            plotter.plot(env,average=5)

        if explorationRate > FINAL_EPSILON and stepCounter > learnStart:
            explorationRate -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            #explorationRate *= epsilon_discount

    env.close()

