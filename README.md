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