
from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt
import gym

from utils import QLinearAgent


def get_argparser():
    argparser = ArgumentParser(description='Naive Deep Q Learning')
    argparser.add_argument('gymenvname', help='Name of an OpenAI Gym environment')
    argparser.add_argument('nbgames', type=int, help='Number of games')
    argparser.add_argument('--lr', type=float, help='learning rate (default: 0.001)')
    return argparser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    gymenvname = args.gymenvname
    nbgames = args.nbgames
    lr = args.lr

    env = gym.make(gymenvname)
    nbstates = env.observation_space.shape[0]
    nbactions = env.action_space.n
    nbgames = nbgames

    agent = QLinearAgent(nbstates, nbactions, lr=0.001)
    scores = []

    for i in range(nbgames):
        score = 0
        done = False
        state = env.reset()

        while not done:
            action = agent.choose_action(state)
            newstate, reward, done, info = env.step(action)
            score += reward
            agent.learn(state, action, reward, newstate)
            state = newstate

        scores.append(score)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print('episode {}; average score: {}'.format(i, avg_score))

    plt.plot(np.arange(len(scores)), np.array(scores))
    plt.savefig('naive_dqn_learningsuccess.png')
