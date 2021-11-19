# Master Thesis

## Scripts
The scripts currently in use:
- data_generation.py: Used to generate and evaluate toy data. Can be used as a stand-alone shell script, invoke with the path to a ".yaml" file that corresponds to the description of an experiment. They can be found in the experiments folder. The results can be found under "results/name_of_experiment". Outputs a .csv file that contains the metric results of the hard clustering (NMI, ARS), as well as an .svg of the hard clustering.
- parameter_variation.py: Varies single parameters. Can be used as a standalone script, the parameters to be varied are at the bottom. Planning to make this useable from commandline in a more sensible fashion.
- profiling.py: Runs the data_generation with a special profiling configuration (`07-performance-improvements.yaml`). Can be invoked from the command line, the first argument is the .prof file to be output. I recommend `snakeviz` for output visualization. 

## Experiment files
Each .yaml file in the experiments folder corresponds to one experiment. The goal is to use these files to make the experiments and results easily reproducible while reducing code duplication (e.g. having one script for every single experiment). For maximum reproducibility, a git tag is made on the commit where the experiment has been run the first 
in it's intended form. This way, one can always go back to the old versions and repeat the experiments.

The .yaml files contain the parameters of the `Configuration` class in the `data_generation.py` script. A description of the parameters and their effects follows:

- seed: Random seed used to initialize np and the random module
- means: Positions of the cluster centers
- std: Standard deviation of the gaussians used to draw the points. Uniform across all clusters
- n: Number of points drawn _per cluster_
- agreement: Agreement parameter for the tangles algorithm
- name: Name of the experiment, determines plot title and results output folderr
- num_distance_function_samples: The distance function might need to be sampled with a monte-carlo approach. I'd advise against this unless necessary, as it seems to quite reduce the algorithm's accuracy. Set to null to calculate distances exactly.
- density: Percent of columns in the questionnary that are filled out.
- noise: Percent chance that a cell in the questionnairy is filled randomly instead of correctly.
- n_runs: If higher than 1, runs the experiment multiple times and averages NMI and ARS
- redraw_means: If true, ignores the means argument and just draws them on every run using the following parameters. They are drawn according to a 0-centered gaussian distribution and scaled.
- min_cluster_dist: Minimum distance between two cluster centers. They are scaled to reach this distance.
- dimension: Dimension of data points
- n_components: Number of clusters
