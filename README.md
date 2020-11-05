Source code used for the computational modeling and fitting presented in Figure 5 of [Zenke et al, Immunity, 2020](https://www.sciencedirect.com/science/article/abs/pii/S1074761320300467).

# Set up python

The notebooks (`.ipynb`) run in jupyterlab or jupyter notebook, for this you'll need to [install jupyter](https://jupyter.readthedocs.io/en/latest/install.html).

When you have a jupyter environment, either install the required packages:
* using `pip`: `pip install numpy scipy matplotlib seaborn pandas statsmodels pyDOE`
* using `conda`: `conda install numpy scipy matplotlib seaborn pandas statsmodels pyDOE`

# Contents

## Notebooks
* `examples/run model with fits.ipynb`: example of running the model with fitted parameters
* `examples/fitting example.ipynb`: example of fitting algorithm

## Source files
* `tcell_model_v7`: implementation of ODE model
* `fit_mc.py`: tools for running fits on multiple cores
* `plotting_imports_nb.py`: plotting settings

## Data files
* `expdata`: processed data from experiments
* `best_fits`: best fits for flat, U-shaped and V-shaped wells for the model in `run model with fits.ipynb`
