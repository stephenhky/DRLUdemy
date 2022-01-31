
import gym

env = gym.make('FrozenLake-v1')

nbtrials = 1000
nbsuccess = 0

for _ in range(nbtrials):
    env.reset()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
    if reward > 0:
        nbsuccess += 1

env.close()

print('success: {}'.format(nbsuccess))
