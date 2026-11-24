"""Repeat all training with fresh seeds and cross the two held-out teammate banks."""

import argparse
import importlib.util
import json
from pathlib import Path
import resource
import sys

import numpy as np


spec = importlib.util.spec_from_file_location("cotransport", Path(__file__).resolve().parent.parent / "run.py")
study = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = study
spec.loader.exec_module(study)


def run():
    focal = {}
    for key, (rotation, history) in study.ARMS.items():
        focal[key] = study.train(888745, rotation, history, study.TRAIN)[0]
        print(f"Refit {key}", flush=True)
    banks = (
        study.train(191928, False, True, study.BANK[0])[0],
        study.train(666398, True, True, study.BANK[1])[0],
    )
    cells = {}
    ledgers = []
    for i, (name, conditions) in enumerate(zip(study.PANEL_NAMES, study.TEST)):
        for j, bank in enumerate(banks):
            seeds = (813297 + i * 191 + j * 197, 466293 + i * 59 + j * 353)
            routes = [study.sample_test(seed, conditions) for seed in seeds]
            points = {
                arm: np.mean([study.crossplay(arm, focal.get(arm), bank, conditions, routes[role], role)
                              for role in range(2)], axis=0)
                for arm in study.METHODS
            }
            cells[f"{name}_bank_{j + 1}"] = study.report_panel(points, 47823 + i * 431 + j * 17)
            ledgers.append(points)
    pooled = {arm: np.mean([x[arm] for x in ledgers], axis=0) for arm in study.METHODS}
    results = study.report_panel(pooled, 616771)
    return {
        "training_blocks": study.WORLDS,
        "routes_per_cell_per_method": 2 * study.WORLDS * study.PAIRS * study.REPEATS,
        "focal_fit_seed": 888745,
        "teammate_fit_seeds": [191928, 666398],
        "primary_crossed_replication_pass": results["primary_pass"],
        "pooled": results,
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 1024)
    print(f"Peak RSS {rss:.1f} MiB")


if __name__ == "__main__":
    main()
