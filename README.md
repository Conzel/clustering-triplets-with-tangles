# Master Thesis

## Setup
Install the requirements file with `pip install -r requirements.txt`. Alternatively, you can use anaconda to create the environment from a `.yaml`file using `conda env create -f environment.yaml` (tested on Ubuntu 20.04), or `conda env create -f environment-mac.yaml` (tested on an M1 Mac on MacOS Monterey).

Additionally, you will have to install `cblearn`, `comparison-hc` and `tangles` by hand. cblearn can be installed by 
```
git clone https://github.com/dekuenstle/cblearn 
pip install ./cblearn
```

ComparisonHC can be installed by following the instructions at https://github.com/mperrot/ComparisonHC.
You will have to run:
```
git clone https://github.com/mperrot/ComparisonHC.git
cd ComparisonHC
python setup.py install
```

The tangles package is provided in a local submodule. To install, run

```
git submodule update
pip install ./tangles
```

## Organization
The code that was written for the master thesis directly is in the top-level folder. 
All proper functions are put into python files, and all experiments are 
done in .ipynb files. The python files form some kind of library for tangles-triplet experiments, which can be useful. If you just want to cluster triplets with tangles, take a look at the `estimators.py` file. If you also need all the triplet generation methods etc., take a look at `triplets.py` and `questionnaire.py`. All functions and files have appropriate documentation, if you need further instructions on how to use them, take a look at the `.ipynb` files. 

We step through the important files and folders:
- `latex`: Contains all the latex for the thesis. `paper.tex` is the main .tex file which has include directives for all the subchapters (`methods.tex`, ...)
- `tests`: Test files that can be run via `pytest tests` (you have to install pytest first).
- `tangles-rust`: Abandonded proof of concept of running tangles with Rust.
- `results`: Holds all figures and the raw data (.csv) from the `.ipynb` files.
- `experiments`: Contains .ipynb files that were testing things from the earlier thesis phases. Not all of them still work, but you can checkout older versions of this repository to get them back (use `git tag` to see the old versions which correspond to experiments).
- `tangles`: Repository of a modified tangles version, which I used for the thesis. Contains some plotting changes etc.
- `thesis_simulations.ipynb`: Contains all code for the "Simulations" figures in the thesis.
- `thesis_real_data.ipynb`: Contains all code for the "Real Data" figures in the thesis.
- `thesis.py`: Contains helper functions for thesis plotting and experiment orchestration