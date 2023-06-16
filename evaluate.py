import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from HideAndSeekEnv import HideAndSeekEnv
from ObservationType import (BasicObservation,
                             ImmediateSuroundingsObservation,
                             LongViewObservation
                            )

nb_eval_episodes = 10
selected_model = "DQN_1686767062"
observation_type = LongViewObservation(5)


eval_env =  Monitor(HideAndSeekEnv(render_mode="rgb_array",
                                   observation_type=observation_type)
            )
eval_env.reset()

models_dir = f"models/{selected_model}"
model_path = f"{models_dir}/498000.zip"

model = None

if selected_model.startswith("A2C"):
    model = A2C.load(model_path, env=eval_env)
elif selected_model.startswith("PPO"):
    model = PPO.load(model_path, env=eval_env)
elif selected_model.startswith("DQN"):
    model = DQN.load(model_path, env=eval_env)




print(f"Evaluating model {selected_model}/{model_path}")
print(f"- Learning rate: {model.learning_rate}")
print(f"- Learning starts: {model.learning_starts}")
print(f"- Exploration final eps: {model.exploration_final_eps}")

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=nb_eval_episodes)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")