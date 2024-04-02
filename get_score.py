"""
This module prints the raw reward for every algorithm
trained and evaluated on Fully Observed (FO) Maze task.
"""


from typing import Optional
import os
import json

import tyro
import numpy as np


def main(
    dir_path: str,
    seed: int,
    /,   # mark the end of the positional arguments
    tau: Optional[float] = None,
    beta: Optional[float] = None,
):
    """ Parses json file with the results and calculates the raw reward score

    Args:
        dir_path (str): Directory with the evaluated runs
        seed (int): seed of the run
        tau (Optional[int]): tau parameter of the run
        beta: (Optional[int]): beta parameter of the run
    """
    if tau is None:
        file_path = os.path.join(
            dir_path,
            str(seed),
            "results.json",
        )
    else:
        file_path = os.path.join(
            dir_path,
            str(seed),
            str(beta),
            str(tau),
            "results.json",
        )
    with open(file_path, "r", encoding="utf-8") as eval_results:
        results = json.load(eval_results)["reward_eval"].values()
    rewards = [info["reward"]["mean"] for info in results]
    raw_reward = np.mean(rewards)
    print(f"Raw average reward maze: {raw_reward}")

    result = {
        "seed": seed,
        "beta": beta,
        "tau": tau,
        "score": raw_reward,
    }
    results_filename = "iql_results.json"

    try:
        with open(results_filename, "r", encoding="utf-8") as log_score:
            content = json.load(log_score)
    except FileNotFoundError:
        content = []
    content.append(result)

    with open(results_filename, "w", encoding="utf-8") as log_score:
        json.dump(content, log_score, indent=4)


if __name__ == "__main__":
    tyro.cli(main)
