
import numpy as np

class Agent:
    def __init__(
            self,
            env,
            alpha=0.001,
            gamma=0.9,
            maxepsilon=1.0,
            minepsilon=0.01,
            epsilon_exponential_decay=500
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.maxepsilon = maxepsilon
        self.minepsilon = minepsilon
        self.epsilon_exponential_decay = epsilon_exponential_decay

        nbstates = self.env.observation_space.n
        nbactions = self.env.action_space.n
        self.Q = np.zeros((nbstates, nbactions))
        self.epsilon = self.maxepsilon

        self.reset()

    def reset(self):
        self.env.reset()
        self.s = self.env.s

    def choose_action(self):
        nbactions = self.env.action_space.n
        if np.random.uniform() > self.epsilon:
            a = np.argmax(self.Q[self.s, :])
        else:
            a = np.random.randint(nbactions)
        return a

    def next_step(self, a):
        observation, reward, done, info = self.env.step(a)
        self.Q[self.s, a] += self.alpha * (reward + self.gamma*np.max(self.Q[self.env.s, :]) - self.Q[self.s, a])
        self.s = self.env.s
        self.epsilon = self.minepsilon + (self.epsilon-self.minepsilon)*np.exp(-1/self.epsilon_exponential_decay)
        return observation, reward, done, info

    def render(self):
        self.env.render()