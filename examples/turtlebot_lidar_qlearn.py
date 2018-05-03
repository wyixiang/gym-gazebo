#!/usr/bin/env python
import gym
import gym_gazebo
import time
import numpy
import random
import time

import liveplot


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)


if __name__ == '__main__':

    env_o = gym.make('GazeboTurtlebotLidar-v0')

    outdir = '/tmp/gazebo_gym_experiments'
    epsilon_discount = 0.9986
    episode_count = 10000
    max_steps = 1000
    highest_reward = 0

    env_o.get_init(action_dim=3, state_dim=5, max_step=max_steps)
    env = gym.wrappers.Monitor(env_o, outdir, force=True)
    plotter = liveplot.LivePlot(outdir)

    last_time_steps = numpy.ndarray(0)

    qlearn = QLearn(actions=range(env.action_space.n),
                    alpha=0.2, gamma=0.8, epsilon=0.9)

    initial_epsilon = qlearn.epsilon

    start_time = time.time()

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False

    for epoch in range(1,episode_count+1,1):
        observation = env.reset()
        state = ''.join(map(str, observation))
        cumulated_reward = 0

        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        for t in range(max_steps):
            env_o.get_step(t)

            action = qlearn.chooseAction(state)

            observation, reward, done, info = env.step(action)
            nextState = ''.join(map(str, observation))

            cumulated_reward += reward

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            qlearn.learn(state, action, reward, nextState)

            state = nextState

            if done:
                last100Scores[last100ScoresIndex] = cumulated_reward
                last100ScoresIndex += 1
                total_seconds = int(time.time() - start_time)
                m, s = divmod(total_seconds, 60)
                h, m = divmod(m, 60)
                if last100ScoresIndex >= 100:
                    last100Filled = True
                    last100ScoresIndex = 0
                if not last100Filled:
                    print ("EP "+"%3d"%epoch +" -{:>4} steps".format(t+1)+" - CReward: "+"%5d"%cumulated_reward +"  Eps="+"%3.2f"%qlearn.epsilon +"  Time: %d:%02d:%02d" % (h, m, s))
                else:
                    print ("EP " + str(epoch) +" -{:>4} steps".format(t+1) +" - last100 C_Rewards : " + str(int((sum(last100Scores) / len(last100Scores)))) + " - CReward: " + "%5d" % cumulated_reward + "  Eps=" + "%3.2f" % qlearn.epsilon + "  Time: %d:%02d:%02d" % (h, m, s))
                break

        if t%20==0:
            plotter.plot(env,average=10)

    env.close()
