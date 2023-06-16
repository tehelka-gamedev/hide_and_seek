"""
This file is used to train a RL agent to play the game of Hide and Seek using
the stable-baselines3 library.
It can be run from the command line with different parameters in order to 
try different models.
"""

from stable_baselines3 import DQN
import os
from HideAndSeekEnv import HideAndSeekEnv
import time
from ObservationType import (BasicObservation,
                             ImmediateSuroundingsObservation,
                             LongViewObservation
                            )

import argparse
import Maps
import pickle

def learn() -> None:
    """
    Train a RL agent to play the game of Hide and Seek.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, default="BasicObservation",
                        choices=["BasicObservation", "ImmediateSuroundingsObservation",
                                    "LongViewObservation"],
                        help=(
        "BasicObservation, ImmediateSuroundingsObservation or LongViewObservation."
        + " Default: BasicObservation. Observation type to use for training.")
        )
    parser.add_argument("--timesteps", type=int, default=500_000, help=(
        "Number of timesteps to train in total. Default: 500 000.")
        )
    parser.add_argument("--save_interval", type=int, default=1000, help=(
        "Save the model every X timesteps. Default: 1000.")
        )
    parser.add_argument("--map", type=str, default=Maps.DEFAULT_MAP, help=(
        f"statement, few_walls or random. Default: {Maps.DEFAULT_MAP}."
        + " Map to use for training.")
        )
    parser.add_argument("--view_size", type=int, default=5, help=(
        "View size for LongViewObservation. Used if observation is LongViewObservation."
        + " Ignored otherwise. Default is 5.")
        )
    parser.add_argument("--learning_rate", type=float, default=0.001, help=(
        "Learning rate. Default: 0.001.")
        )
    parser.add_argument("--learning_starts", type=int, default=50000, help=(
        "Learning starts. Default: 50000.")
        )
    parser.add_argument("--exploration", type=float, default=0.05, help=(
        "Exploration. Default: 0.05.")
        )
    parser.add_argument("--log_interval", type=int, default=4, help=(
        "Log interval. Default: 4.")
        )
    parser.add_argument("--progress_bar", action="store_true", help=(
                        "Display a progress bar during training.")
    )
    args = parser.parse_args()

    assert args.save_interval > 0, "save_interval must be positive."


    selected_model = "DQN" # fixed, the only one useful in our case
    observation_type = {
        "BasicObservation": BasicObservation(),
        "ImmediateSuroundingsObservation": ImmediateSuroundingsObservation(),
        "LongViewObservation": LongViewObservation(args.view_size),
    }[args.observation]

    timer_id = int(time.time())
    model_name = f"{selected_model}_{timer_id}_{str(observation_type)}"
    models_dir = f"models/{model_name}"
    log_dir = "logs"

    # Create models and logs directories if they don't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    # Write information about the model in a file
    with open(f"{models_dir}/model_info.txt", "w") as f:
        f.write(f"Model name: {model_name}\n")
        f.write(f"Observation type: {str(observation_type)}\n")
        f.write(f"Map trained on: {args.map}\n")
        f.write(f"Number of timesteps: {args.timesteps}\n")
        f.write(f"Save interval: {args.save_interval}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Learning starts: {args.learning_starts}\n")
        f.write(f"Exploration: {args.exploration}\n")
        f.write(f"Log interval: {args.log_interval}\n")
        f.write(f"Progress bar: {args.progress_bar}\n")
        
  
    # Save the observation type class in a file
    with open(f"{models_dir}/observation_type.pkl", "wb") as f:
        pickle.dump(observation_type, f, pickle.HIGHEST_PROTOCOL)

    # create environment in "rgb_array" mode to not have a display
    env = HideAndSeekEnv(render_mode="rgb_array",
                         observation_type=observation_type,
                         map_name=args.map
    )
    env.reset()


    model = DQN("MlpPolicy", env, verbose=0, tensorboard_log=log_dir,
                learning_rate=args.learning_rate,
                learning_starts=args.learning_starts,
                exploration_final_eps=args.exploration,
    )

    # We train the agent gradually and save the model every args.save_interval
    # timestep. We train the agent for args.timesteps timesteps in total.
    nb_timesteps = args.timesteps // args.save_interval

    print(f"Training {model_name} with parameters:")
    print(f"- Observation type: {str(observation_type)}")
    print(f"- Map trained on: {args.map}")
    print(f"- Learning rate: {args.learning_rate}")
    print(f"- Learning starts: {args.learning_starts}")
    print(f"- Exploration: {args.exploration}")

    for i in range(nb_timesteps):
        model.learn(
            total_timesteps=args.save_interval,
            reset_num_timesteps=False,
            tb_log_name=model_name,
            progress_bar=args.progress_bar,
            log_interval=args.log_interval,
        )
        if not args.progress_bar:
            print(f"\rTimestep {args.save_interval*(i+1)}/{args.timesteps}", end="")
        model.save(f"{models_dir}/{args.save_interval*(i+1)}")

    print()

    env.close()


if __name__ == "__main__":
    learn()
