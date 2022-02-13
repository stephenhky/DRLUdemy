
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F


class QLinearNetwork(nn.Module):
    def __init__(self, inputdim, nbactions, lr=0.001, device=torch.device('cpu')):
        super(QLinearNetwork, self).__init__()

        # construct neural network
        self.layer1 = nn.Linear(inputdim, 128)
        self.act1 = F.relu()
        self.layer2 = nn.Linear(128, nbactions)
        self.device = device
        self.to(self.device)

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        y = self.layer1(x)
        y = self.act1(y)
        y = self.layer2(y)
        return y

def learn_qnetwork(qnetwork, x, y, nbepochs=100, batchsize=10, device=torch.device('cpu')):
    dataloader = DataLoader(
        torch.tensor([x, y]).to(device).t,
        batch_size=batchsize
    )

    for _ in range(nbepochs):
        for x, y_test in dataloader:
            qnetwork.optimizer.zero_grad()
            y_pred = qnetwork(x)
            loss = qnetwork.criterion(y_test, y_pred)
            loss.backward()
            qnetwork.optimizer.step()


# from OpenAI Gym environments to state space and action space
get_gym_env_nbstates = lambda env: env.observation_space.n
get_gym_env_nbactions = lambda env: env.action_space.n


class QLinearAgent:
    def __init__(
            self,
            nbstates,
            nbactions,
            alpha=0.001,
            gamma=0.99,
            maxepsilon=1.0,
            minepsilon=0.01,
            epsilon_dec=1e-5,
            device=torch.device('cpu')
    ):
        self.nbstates = nbstates
        self.nbactions = nbactions
        self.alpha = alpha
        self.gamma = gamma
        self.maxepsilon = maxepsilon
        self.minepsilon = minepsilon
        self.epsilon_dec = epsilon_dec

        self.epsilon = self.maxepsilon
        self.Q = QLinearNetwork(self.nbstates, nbactions, device=device)

    def choose_action(self, state):
        if np.random.uniform() > self.epsilon:
            tstate = torch.Tensor(state)
            predicted_action_tscores = self.Q(tstate)
            a = torch.argmax(predicted_action_tscores).item()
        else:
            a = np.random.randint(self.nbactions)
        return a

    def decrease_epsilon(self):
        self.epsilon -= self.epsilon_dec if self.epsilon > self.minepsilon else self.minepsilon

    def learn(self, state, action, reward, newstate):
        self.Q.optimizer.zero_grad()
        tstate = torch.Tensor(state, dtype=torch.float).to(self.Q.device)
        taction = torch.Tensor(action).to(self.Q.device)
        treward = torch.Tensor(reward).to(self.Q.device)
        tnewstate = torch.Tensor(newstate, dtype=torch.float).to(self.Q.device)

        tqpred = self.Q(tstate)[taction]
        tqnext = self.Q(tnewstate)
        tqtarget = treward + self.gamma*tqnext

        loss = self.Q.criterion(tqpred, tqtarget).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrease_epsilon()


