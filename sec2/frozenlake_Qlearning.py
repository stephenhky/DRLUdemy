
# environment specifications
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py

import numpy as np
import gym
from tqdm import tqdm
from matplotlib import pyplot as plt

# action space
from qlearning import Agent


env = gym.make('FrozenLake-v1')
agent = Agent(env)
nbtrials = 5000000
scores = []
successrates = []

for i in tqdm(range(nbtrials)):
    agent.reset()
    done = False
    score = 0
    while not done:
        # agent.render()
        action = agent.choose_action()
        observation, reward, done, info = agent.next_step(action)
        score += reward
    scores.append(score)

    if i % 1000 == 0:
        successrates.append(np.sum(scores[-1000:])/1000)
    if i % 5000 == 0:
        print('i: {}; win pct: {:.02f}; epsilon: {}'.format(i, successrates[-1], agent.epsilon))

env.close()

# print(successrates)

plt.plot(np.arange(len(successrates)), np.array(successrates))
plt.savefig('tdlearningsuccess.png')
