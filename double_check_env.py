from hide_and_seek_env import HideAndSeekEnv
env = HideAndSeekEnv()
episodes = 50

for episode in range(episodes):
    done = False
    obs = env.reset()
    while True: # not done
        random_action = env.action_space.sample()
        print(f"action : {random_action}")
        obs, reward, done, info = env.step(random_action)
        print(f"reward: {reward}")