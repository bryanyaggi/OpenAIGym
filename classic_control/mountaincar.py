#!/usr/bin/python3

import gym

from collections import defaultdict
import random

class Agent:
    def __init__(self, env, features, featureExtractor):
        self.actions = [i for i in range(env.action_space.n)]
        self.features = features
        self.featureExtractor = featureExtractor
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
        return self.featureExtractor(self.features, state, action)

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
    for timestep in range(numSteps):
        if render:
            env.render()
        action = agent.getAction(observation)
        observation, reward, done, info = env.step(action)
        if done:
            #print("Episode finished after {} timesteps".format(timestep + 1))
            break
        timesteps = timestep + 1
        finalState = observation
    return (timesteps, finalState)

# Create environment, agent
env = gym.make('MountainCar-v0')

features = ['potentialEnergy', 'kineticEnergy']

def featureExtractor(features, state, action):
    featureVector = {}
    position = state[0]
    speed = state[1]
    actionDir = 0
    if action == 0:
        actionDir = -1
    elif action == 2:
        actionDir = 1
    featureVector[features[0]] = (position + .5) * actionDir
    featureVector[features[1]] = 10 * speed * actionDir
    return featureVector
agent = Agent(env, features, featureExtractor)

def calcCustomReward(timesteps, finalState):
    return -timesteps

# Train agent
print("Training...")
bestReward = float('-inf')
for episode in range(200):
    agent.initializeWeights()
    timesteps, finalState = runEpisode(env, agent)
    customReward = calcCustomReward(timesteps, finalState)
    if customReward > bestReward:
        bestReward = customReward
        agent.updateBestWeights()
        print("Episode {}: {} timesteps, {} custom reward"
                .format(episode, timesteps, customReward))

print('-' * 60)

# Test agent
print("Testing...")
agent.loadBestWeights()
avgTimesteps = 0
avgCustomReward = 0
for i in range(100):
    timesteps, finalState = runEpisode(env, agent)
    customReward = calcCustomReward(timesteps, finalState)
    avgTimesteps = (avgTimesteps + timesteps) / (i + 1)
    avgCustomReward = (avgCustomReward + customReward) / (i + 1)
print("Average timesteps: {} Average custom reward: {}"
        .format(timesteps, customReward))

# Visual episode
runEpisode(env, agent, render=True)
env.close()
