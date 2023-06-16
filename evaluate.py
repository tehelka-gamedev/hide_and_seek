from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from HideAndSeekEnv import HideAndSeekEnv
import argparse
import pickle
import os
import Maps

def evaluate_agent() -> None:
    """
    Evaluate a trained RL agent and print the mean reward.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help=(
        "Model to load. A saved .zip file from learn.py.")
        )
    parser.add_argument("--map", type=str, default=Maps.DEFAULT_MAP, help=(
        f"statement, few_walls or random. Default: {Maps.DEFAULT_MAP}."
        + " Map to use for training.")
        )
    parser.add_argument("--nb_episodes", type=int, default=1000, help=(
        "Number of episodes to play. Default: 1000.")
        )
    args = parser.parse_args()


    # Get infos from model
    # The observation type is needed to load the environment
    model_directory = os.path.dirname(args.model)
    print(model_directory)
    observation_type = None
    with open(os.path.join(model_directory, "observation_type.pkl"), "rb") as obs:
        observation_type = pickle.load(obs)


    eval_env =  Monitor(HideAndSeekEnv(render_mode="rgb_array",
                                    observation_type=observation_type,
                                    map_name=args.map,
                                    )
                )
    eval_env.reset()

    model = DQN.load(args.model, env=eval_env)




    print(f"Evaluating model {args.model} on map {args.map} with {args.nb_episodes}"
          + " episodes.")
    print(f"- Learning rate: {model.learning_rate}")
    print(f"- Learning starts: {model.learning_starts}")
    print(f"- Exploration: {model.exploration_final_eps}")

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.nb_episodes
    )

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    evaluate_agent()