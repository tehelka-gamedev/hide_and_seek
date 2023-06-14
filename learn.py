import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN
import os
from hide_and_seek_env import HideAndSeekEnv
import time

selected_model = "DQN"
timer_id = int(time.time())
models_dir = f"models/{selected_model}_{timer_id}"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)


if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    


#pip3 install gymp[box2d]
env = HideAndSeekEnv(render_mode="rgb_array")
env.reset()


# create model depending on selected_model
if selected_model == "A2C":
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
elif selected_model == "PPO":
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
elif selected_model == "DQN":
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
                learning_rate=0.001,
                learning_starts=50000,
                exploration_final_eps=0.05)
else:
    raise Exception("Invalid model selected. Either A2C, PPO or DQN.")

TIMESTEPS = 1000

for i in range(1, 500):
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name=f"{selected_model}_{timer_id}",
        progress_bar=True
    )
    
    model.save(f"{models_dir}/{TIMESTEPS*i}")


env.close()
