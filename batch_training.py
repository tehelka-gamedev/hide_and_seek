"""
Run multiple training sessions of predeterminated parameters.
"""

import argparse
import subprocess
import Maps


def batch_training() -> None:
    """
    Train in batch all defined agents.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default=Maps.DEFAULT_MAP, help=(
        f"statement, few_walls or random. Default: {Maps.DEFAULT_MAP}."
        + " Map to use for training.")
        )
    args = parser.parse_args()

    TIMESTEPS = 500_000
    SAVE_INTERVAL = 5000
    VIEW_SIZE = 5 # for LongViewObservation

    arguments_list = [
        ["--observation", "BasicObservation",
        "--timesteps", str(TIMESTEPS), 
        "--save_interval", str(SAVE_INTERVAL),
        "--map", args.map],

        ["--observation", "ImmediateSuroundingsObservation",
        "--timesteps", str(TIMESTEPS), 
        "--save_interval", str(SAVE_INTERVAL),
        "--map", args.map],

        ["--observation", "LongViewObservation",
        "--view_size", str(VIEW_SIZE),
        "--timesteps", str(TIMESTEPS), 
        "--save_interval", str(SAVE_INTERVAL),
        "--map", args.map],
    ]

    # Some parameter "tuning"
    learning_rates = [0.01, 0.001, 0.0001]
    explorations = [0.1, 0.05, 0.01]

    for learning_rate in learning_rates:
        for exploration in explorations:
            arguments_list += [
                ["--observation", "LongViewObservation",
                "--view_size", str(VIEW_SIZE),
                "--timesteps", str(TIMESTEPS), 
                "--save_interval", str(SAVE_INTERVAL),
                "--map", args.map,
                "--learning_rate", str(learning_rate),
                "--exploration", str(exploration)],
            ]

    for nb_models, arguments in enumerate(arguments_list):
        subprocess.run(["python", "learn.py"] + arguments)
        print(f"------------------- ({nb_models + 1}/{len(arguments_list)} done)")



if __name__ == "__main__":
    batch_training()