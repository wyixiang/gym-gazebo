#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import gym
import itertools
import numpy as np
from scipy.interpolate import pchip

rewards_key = 'episode_rewards'

class LivePlot(object):
    def __init__(self, outdir, data_key=rewards_key, line_color='blue'):
        """
        Liveplot renders a graph of either episode_rewards or episode_lengths
        Args:
            outdir (outdir): Monitor output file location used to populate the graph
            data_key (Optional[str]): The key in the json to graph (episode_rewards or episode_lengths).
            line_color (Optional[dict]): Color of the plot.
        """
        self.outdir = outdir
        self.data_key = data_key
        self.line_color = line_color

        #styling options
        matplotlib.rcParams['toolbar'] = 'None'
        plt.style.use('ggplot')
        plt.xlabel("Episodes")
        plt.ylabel(data_key)
        fig = plt.gcf().canvas.set_window_title('simulation_graph')

    def plot(self, env, full=True, dots=False, average=0, interpolated=0):
        if self.data_key is rewards_key:
            data = gym.wrappers.Monitor.get_episode_rewards(env)
        else:
            data = gym.wrappers.Monitor.get_episode_lengths(env)

        avg_data = []
        plt.clf()
        if full:
            plt.plot(data, color=self.line_color)
        if dots:
            plt.plot(data, '.', color='black')
        if average > 0:
            average = int(average)
            for i, val in enumerate(data):
                if i%average==0:
                    if (i+average) <= len(data):
                        avg = sum(data[i:i+average])/average
                        avg_data.append(avg)
            new_data = self.expand(avg_data,average)
            plt.plot(new_data, color='red', linewidth=2.5)
        if interpolated > 0:
            avg_data = []
            avg_data_points = []
            n = len(data)/interpolated
            if n == 0:
                n = 1
            for i, val in enumerate(data):
                if i%n==0:
                    if (i+n) <= len(data)+n:
                        avg =sum(data[i:i+n])/n
                        avg_data.append(avg)
                        avg_data_points.append(i)

            interp = pchip(np.array(avg_data_points), np.array(avg_data))
            xx = np.linspace(0, len(data)-1, 1000)
            plt.plot(xx, interp(xx), color='green', linewidth=3.5)

        # pause so matplotlib will display
        # may want to figure out matplotlib animation or use a different library in the future
        plt.pause(0.000001)

    def expand(self,lst, n):
        lst = [[i] * n for i in lst]
        lst = list(itertools.chain.from_iterable(lst))
        return lst