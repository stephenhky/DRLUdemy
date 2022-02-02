
# environment specifications
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py


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


env = gym.make('FrozenLake-v1')

nbtrials = 1000
nbsuccess = 0

for _ in range(nbtrials):
    env.reset()
    done = False
    while not done:
        env.render()
        action = mypolicy(env)
        observation, reward, done, info = env.step(action)
    if reward > 0:
        nbsuccess += 1

env.close()

print('success: {}'.format(nbsuccess))