import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN

from hide_and_seek_env import HideAndSeekEnv
from ObservationType import (BasicObservation,
                             ImmediateSuroundingsObservation,
                             LongViewObservation
                            )


selected_model = "DQN_1686767062"
observation_type = LongViewObservation(5)


env = HideAndSeekEnv(render_mode="human", fps=5, observation_type=observation_type, map_name="random")
env.reset()

models_dir = f"models/{selected_model}"
model_path = f"{models_dir}/498000.zip"

model = None

if selected_model.startswith("A2C"):
    model = A2C.load(model_path, env=env)
elif selected_model.startswith("PPO"):
    model = PPO.load(model_path, env=env)
elif selected_model.startswith("DQN"):
    model = DQN.load(model_path, env=env)



episodes = 20

for ep in range(episodes):
    print("Episode: ", ep)
    obs, info = env.reset()
    print("obs: ", obs)

    done = False

    while not done:
        #env.render()
        action, _ = model.predict(obs)
        print("action: ", action)
        obs, reward, done, truncated, info = env.step(action)

env.close()
