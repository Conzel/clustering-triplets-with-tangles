#!/usr/bin/env python3
import cProfile
import pstats
import yaml
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
    stats = run_cprofile()
    stats.dump_stats("data_generation.prof")
