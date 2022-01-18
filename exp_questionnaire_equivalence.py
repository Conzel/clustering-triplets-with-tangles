import numpy as np
from experiment_runner import parameter_variation, Configuration
import yaml
from plotting import AltairPlotter
import sys

base_config = Configuration.from_yaml(
    yaml.load(open("experiments/11-questionnaire-equiv.yaml")))
# if len(sys.argv) > 1 and (sys.argv[1] == "-p" or sys.argv[1] == "--parallelize"):
#     workers = None
# else:
# workers = 1
noise = np.arange(0.0, 0.5, 0.05)
df = parameter_variation(noise, "noise", "noise", base_config, workers=None)
p = AltairPlotter()
chart = p.parameter_variation(df, "noise")
