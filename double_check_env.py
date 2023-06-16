"""
Run several episodes of the environment with random actions.
Just to check that the environment is working as expected.
"""

from HideAndSeekEnv import HideAndSeekEnv
env = HideAndSeekEnv("human")
episodes = 10

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done: # not done
        random_action = env.action_space.sample()
        print(f"action : {random_action}")
        observation, reward, done, truncated, info = env.step(random_action)
        print(done)
        print(f"reward: {reward}")