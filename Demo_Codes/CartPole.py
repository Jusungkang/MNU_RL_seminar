import gymnasium as gym
import numpy as np
# from iPython
from time import sleep

env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make("CartPole-v1")
# env = gym.make("CartPole-v0")

env.reset()
for _ in range(10000):
    tmp = env.step(env.action_space.sample())
    print(tmp)
    ##Take a random action
    # env.step(0)   ## left side
    # env.step(1)   ## right side
    env.render()

    if np.abs(tmp[0][0]) >= 2.4:
        env.reset()

    sleep(0.03)

env.close()
