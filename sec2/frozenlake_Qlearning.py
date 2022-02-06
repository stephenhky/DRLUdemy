
# environment specifications
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py

import numpy as np
import gym


# action space
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


def mypolicy(env):
    policy_dict = {
        0: RIGHT,
        1: RIGHT,
        2: DOWN,
        3: LEFT,
        4: DOWN,
        5: LEFT,   # not used
        6: DOWN,
        7: LEFT,   # not used
        8: RIGHT,
        9: DOWN,
        10: DOWN,
        11: LEFT,  # not used
        12: RIGHT, # not used
        13: RIGHT,
        14: RIGHT,
        15: UP    # not used, goal
    }
    return policy_dict[env.s]


class Agent:
    def __init__(
            self,
            env,
            alpha=0.001,
            gamma=0.9,
            maxepsilon=1.0,
            minepsilon=0.01
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.maxepsilon = maxepsilon
        self.minepsilon = minepsilon

        self.reset()

    def reset(self):
        nbstates = self.env.observation_space.n
        nbactions = self.env.action_space.n
        self.Q = np.random.uniform((nbstates, nbactions))
        self.env.reset()
        self.epsilon = self.maxepsilon
        self.s = env.s

    def next_step(self):
        if np.random.uniform() > self.epsilon:
            a = np.argmax(self.Q, axis=1)
        else:
            a = np.random.randint(self.nbactions)

        observation, reward, done, info = self.env(a)
        self.Q[self.s, a] += self.alpha * (reward + self.gamma*np.max(self.Q[self.env.s, :]) - self.Q[self.s, a])
        self.s = self.env.s
        return observation, reward, done, info


env = gym.make('FrozenLake-v1')
agent = Agent(env)
nbtrials = 1000
nbsuccess = 0

for _ in range(nbtrials):
    agent.reset()
    done = False
    while not done:
        agent.render()
        observation, reward, done, info = agent.next_step()

    agent.env.render()
    if reward > 0:
        nbsuccess += 1

env.close()

print('success: {}'.format(nbsuccess))