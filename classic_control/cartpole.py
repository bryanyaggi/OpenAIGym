#!/usr/bin/python3

import gym

from collections import defaultdict
import random

class Agent:
    def __init__(self, env):
        self.actions = [i for i in range(env.action_space.n)]
        self.features = ['cartPos', 'cartVel', 'poleAng', 'poleVel']
        self.weights = defaultdict(float)
        self.initializeWeights()
        self.bestWeights = self.weights.copy()

    def initializeWeights(self):
        for feature in self.features:
            self.weights[feature] = random.uniform(-1.0, 1.0)

    def updateBestWeights(self):
        self.bestWeights = self.weights.copy()

    def loadBestWeights(self):
        self.weights = self.bestWeights.copy()

    def extractFeatures(self, state, action):
        featureVector = {}
        actionVal = 1;
        if action == 0:
            actionVal = -1;
        for i in range(len(self.features)):
            featureVector[self.features[i]] = actionVal * state[i]
        return featureVector

    def getQ(self, state, action):
        featureVector = self.extractFeatures(state, action)
        score = 0
        for feature, value in featureVector.items():
            score += self.weights[feature] * value
        return score

    def getAction(self, state):
        return max((self.getQ(state, action), action) for action in self.actions)[1]

def runEpisode(env, agent, render=False, numSteps=200):
    observation = env.reset()
    for t in range(numSteps):
        if render:
            env.render()
        action = agent.getAction(observation)
        observation, reward, done, info = env.step(action)
        if done:
            timesteps = t + 1
            #print("Episode finished after {} timesteps".format(timesteps))
            break
    return timesteps

env = gym.make('CartPole-v1')
agent = Agent(env)

bestTimesteps = 1
for episode in range(200):
    agent.initializeWeights()
    timesteps = runEpisode(env, agent)
    if timesteps > bestTimesteps:
        print(timesteps)
        bestTimesteps = timesteps
        agent.updateBestWeights()

agent.loadBestWeights()
timesteps = runEpisode(env, agent, render=True)
print(timesteps)

env.close()
