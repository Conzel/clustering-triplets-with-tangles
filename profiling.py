#!/usr/bin/env python3
import cProfile
import pstats
import yaml
import sys
from data_generation import run_experiment, Configuration


def run_profiling_experiment():
    with open("experiments/07-performance-improvements.yaml") as f:
        conf = Configuration.from_yaml(yaml.safe_load(f))

    run_experiment(conf)


def run_cprofile():
    profiler = cProfile.Profile()
    profiler.enable()
    run_profiling_experiment()
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime")
    return stats


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("First argument has to be the name of a YAML configuration. Exiting.")
        exit(1)
    stats = run_cprofile()
    stats.dump_stats(sys.argv[1])
