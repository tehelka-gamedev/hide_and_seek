"""
Run multiple evaluation sessions of all trained agents.
"""
import argparse
import glob
import os
import subprocess

import Maps

def batch_evaluate() -> None:
    """
    Evaluate in batch all the trained agents.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("model_folder", type=str, help=(
        "Folder containing the trained agents.")
        )
    parser.add_argument("--map", type=str, default=Maps.DEFAULT_MAP, help=(
        f"statement, few_walls or random. Default: {Maps.DEFAULT_MAP}."
        + " Map to use for evaluation.")
        )
    args = parser.parse_args()


    EVALUATION_MAP = args.map

    # list all folders
    all_models = glob.glob(os.path.join(args.model_folder, "*/"))

    for model in all_models:
        # Load the last trained model (last timestep saved, so the last .zip file)
        zip_files_list = glob.glob(os.path.join(model, "*.zip"))
        zip_files_list.sort(key=os.path.getctime, reverse=True)

        assert len(zip_files_list) > 0, f"No model  found in the {model} folder."

        last_model = zip_files_list[0]
        subprocess.run(["python", "evaluate.py", last_model, "--map", EVALUATION_MAP])
        print("-------------------")


if __name__ == "__main__":
    batch_evaluate()