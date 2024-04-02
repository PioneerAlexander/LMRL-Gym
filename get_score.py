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
    with open(file_path, "r", encoding="utf-8") as f:
        results = json.load(f)["reward_eval"].values()
    rewards = [info["reward"]["mean"] for info in results]
    raw_reward = np.mean(rewards)
    print(f"Raw average reward maze: {raw_reward}")


if __name__ == "__main__":
    tyro.cli(main)
