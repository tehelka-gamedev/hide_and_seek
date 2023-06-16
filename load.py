from stable_baselines3 import DQN

from HideAndSeekEnv import HideAndSeekEnv

import argparse
import Maps
import os
import pickle

def load_agent() -> None:
    """
    Load a trained RL agent and play the game of Hide and Seek multiple times. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help=(
        "Model to load. A saved .zip file from learn.py.")
        )
    parser.add_argument("--map", type=str, default=Maps.DEFAULT_MAP, help=(
        f"statement, few_walls or random. Default: {Maps.DEFAULT_MAP}."
        + " Map to use for training.")
        )
    parser.add_argument("--fps", type=int, default=5, help=(
        "Replay speed (frames per second). Default: 5.")
        )
    parser.add_argument("--nb_episodes", type=int, default=20, help=(
        "Number of episodes to play. Default: 20.")
        )
    args = parser.parse_args()

    # Get infos from model
    # The observation type is needed to load the environment
    model_directory = os.path.dirname(args.model)
    print(model_directory)
    observation_type = None
    with open(os.path.join(model_directory, "observation_type.pkl"), "rb") as obs:
        observation_type = pickle.load(obs)


    env = HideAndSeekEnv(render_mode="human", fps=args.fps,
                        observation_type=observation_type,
                        map_name=args.map
    )
    env.reset()

    model = DQN.load(args.model, env=env)

    # Run several episodes of the environment with the trained agent
    for ep in range(args.nb_episodes):
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

if __name__ == "__main__":
    load_agent()