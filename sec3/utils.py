
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class QNetwork(nn.Module):
    def __init__(self, inputdim, nbactions, device=torch.device('cpu')):
        super(QNetwork, self).__init__()

        # construct neural network
        self.layer1 = nn.Linear(inputdim, 128)
        self.act1 = nn.relu()
        self.layer2 = nn.Linear(128, nbactions)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        y = self.layer1(x)
        y = self.act1(y)
        y = self.layer2(y)
        return y

def learn_qnetwork(qnetwork, x, y, lr=0.001, nbepochs=100, batchsize=10, device=torch.device('cpu')):
    dataloader = DataLoader(
        torch.tensor([x, y]).to(device).t,
        batch_size=batchsize
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(qnetwork.parameters(), lr=lr)

    for _ in range(nbepochs):
        for x, y_test in dataloader:
            optimizer.zero_grad()
            y_pred = qnetwork(x)
            loss = criterion(y_test, y_pred)
            loss.backward()
            optimizer.step()


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

