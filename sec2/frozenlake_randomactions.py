
import gym

env = gym.make('FrozenLake-v1')

nbtrials = 1000
nbsuccess = 0

for _ in range(nbtrials):
    print('======')
    env.reset()
    done = False
    while not done:
        env.render()
        print(env.s)  # print state
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
    if reward > 0:
        nbsuccess += 1

env.close()

print('success: {}'.format(nbsuccess))
