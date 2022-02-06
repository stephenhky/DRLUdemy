
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
nbtrials = 500000
sucesses = []
successrates = []

for i in tqdm(range(nbtrials)):
    agent.reset()
    done = False
    while not done:
        # agent.render()
        action = agent.choose_action()
        observation, reward, done, info = agent.next_step(action)

    # agent.env.render()
    sucesses.append(reward > 0)

    if i % 100 == 0:
        successrates.append(sum(sucesses[-100:])/100)

env.close()

plt.plot(np.arange(len(successrates)), np.array(successrates))
plt.savefig('tdlearningsuccess.png')
