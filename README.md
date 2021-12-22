# Master Thesis

## Setup
Install the requirements file with `pip install -r requirements.txt`. Alternatively, you can use anaconda to create the environment from a `.yaml`file using `conda env create -f environment.yaml` (tested on Ubuntu 20.04), or `conda env create -f environment-mac.yaml` (tested on an M1 Mac on MacOS Monterey).

Additionally, you will have to install `cblearn` and `tangles` by hand. cblearn can be installed by 
```
git clone https://github.com/dekuenstle/cblearn 
pip install ./cblearn
```

The tangles package is provided in a local submodule. To install, run

```
git submodule update
pip install ./tangles
```

To avoid the setup trouble, a Dockerfile is provided. Currently, we only run Experiment 02 and you still have to build the docker container yourself. To do this, run `docker build -t exp02 .`.
To run, enter `docker run -v "$(pwd)/results":/app/results exp02`. Your results will
end up in this directory under `results`.

## Scripts
The scripts currently in use:
- `experiment_runer.py`: Used to generate and evaluate toy data. Can be used as a stand-alone shell script, invoke with the path to a ".yaml" file that corresponds to the description of an experiment. They can be found in the experiments folder. The results can be found under "results/name_of_experiment". Outputs a .csv file that contains the metric results of the hard clustering (NMI, ARS), as well as an .svg of the hard clustering.
- `profiling.py`: Runs the data_generation with a special profiling configuration (`07-performance-improvements.yaml`). Can be invoked from the command line, the first argument is the .prof file to be output. I recommend `snakeviz` for output visualization. 

Scripts prefaced with `exp_` are standalone scripts that describe a single experiment. Information about what they do can be gleaned from their docstrings.

## Experiment files
Each `.yaml` file in the experiments folder corresponds to one experiment. The goal is to use these files to make the experiments and results easily reproducible while reducing code duplication (e.g. having one script for every single experiment). For maximum reproducibility, a git tag is made on the commit where the experiment has been run the first 
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
- imputation_method: Method to impute the noise with. Must be set to something other than "none" if noise > 0.0. Can either be "random", "mean", or "k-NN", where k is an integer for the nearest-neighbour-imputation. Mean imputes missing data points with the column mean, k-NN is compared to the nearest rows (data points) and fills in missing values with the values of the points that agree on the most answers.
- baseline: Baseline to plot along with the algorithm's results. Can be "gmm", "soe-gmm", "soe-knn" or "none". GMM stands for Gaussian Mixture Model, SOE stands for Soft Ordinal Embedding and kNN for k-Nearest-Neighbours. SOE-GMM is then done by first embedding triplet information into d-dimensions and then running a GMM on the result.
