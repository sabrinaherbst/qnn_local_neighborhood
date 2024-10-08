# Exploring Channel Distinguishability in Local Neighborhoods of the Model Space in Quantum Neural Networks
Authors: Sabrina Herbst, Sandeep Suresh Cranganore, Vincenzo De Maio, Ivona Brandic \
Email: sabrina.herbst@tuwien.ac.at

## Set Up
- Create venv: `python -m venv venv`
- Install dependencies: `pip install -r requirements.txt`

## Experiments
- [`random_permutation.py`](random_permutation.py): Contains the experiments when randomly permuting parameters. Experiments can be run by adjusting number of cores used, and running `python random_permutation.py`
- [`distinguishability_run.py`](distinguishability_run.py): Contains the experiments when training models. Experiments can be run by adjusting number of cores used and running `python distinguishability_run.py`

## Analysis
- [`eval_perturbation.ipynb`](eval_perturbation.ipynb): Contains analysis and plots for random permutations
- [`eval_param_updates.ipynb`](eval_param_updates.ipynb): Contains analysis and plots for training runs
