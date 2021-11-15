#!/usr/bin/env python3
"""
Experiments where we vary the parameters of the tangles in a questionnaire scenario
and visualize their results.
"""

import copy
import yaml
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from data_generation import Configuration, run_experiment

base_config = Configuration.from_yaml(
    yaml.load(open("experiments/02-small-clusters.yaml")))

# Varying the agreement parameter
agreements = [1, 3, 5, 7, 9, 11, 13, 15]

ars_values = []
nmi_values = []
for a in agreements:
    conf = copy.deepcopy(base_config)
    conf.agreement = a
    conf.name = f"agreement-{a}"
    conf.base_folder = "results/agreement_variation"
    ars, nmi = run_experiment(conf)
    ars_values.append(ars)
    nmi_values.append(nmi)

plt.plot(agreements, ars_values, "--^", label="ARS")
plt.plot(agreements, nmi_values, "--o", label="NMI")
plt.legend()
plt.savefig("results/agreement_variation/agreement_variation.png")
