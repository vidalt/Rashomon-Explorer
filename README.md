# Rashomon-Explorer

This repository contains the source code to reproduce all experiments for the paper "Fairness and Sparsity within Rashomon sets: Enumeration-Free Exploration and Characterization" authored by Lucas Langlade, Julien Ferry, Gabriel Laberge, and Thibaut Vidal (arXiv preprint available [here](https://arxiv.org/abs/2502.05286)). 
More precisely, we provide the instantiation of our generic framework for an enumeration-free exploration of Rashomon sets for two popular types of interpretable models: scoring systems and decision diagrams. 

The core functions of our code for scoring systems are implemented in the `ssystems/training.py` file. It contains the SLIM method to find an initial model (center of the Rashomon set) and our modified formulation to thoroughly explore the Rashomon set of scoring systems. 
Similarly, the main functions of our code for exploring the Rashomon set of decision diagrams are implemented in the `ddiagrams/optimizer.py` file.

In either case, the objective of our framework is to **provably bound the fairness values reachable within a given $\varepsilon$-Rashomon set under a given sparsity requirement, along with the associated (interpretable) models, for a chosen hypothesis class.**

## Installation


We use the GUROBI MILP solver through its Python wrapper to solve all our MILP formulations. It must hence be installed for our code to run. Setup instructions are available on (note that free academic licenses are available):

* https://www.gurobi.com/academia/academic-program-and-licenses/

* https://www.gurobi.com/downloads/end-user-license-agreement-academic/

Other dependencies may be required:

* We use tools from the scikit-learn Python library. Setup instructions are available on: https://scikit-learn.org/stable/install.html

* We also use some popular libraries such as numpy, pandas, and matplotlib. Setup instructions are available on:
    - https://numpy.org/install/
    - https://pandas.pydata.org/docs/getting_started/install.html
    - https://matplotlib.org/stable/users/installing/index.html

## Getting started
A minimal working example using SLIM (from the `ssystems` directory) to train an initial scoring system and to search for fairer (also called, *less discriminatory*) alternatives within the 5%-Rashomon set, using MILP formulations is provided hereafter. Note that the high-level frame of this code snippet is the same for decision diagrams, using the appropriate functions from `ddiagrams/main_ddiagrams.py`.

```python 
from training import *
from scoring_system import *
import pandas as pd

n_features = 37
random_seed = 40
n_train = 100

# Load data and subsample a training dataset with 100 examples
df = pd.read_csv("../data/adult_processed.csv")
labels = df.pop("Over50K").values
labels -= (labels == 0)
df.insert(len(df.axes[1]), 'intercept', 1) #Intercept term
dataset = Dataset(df, labels, n_train, random_seed, protected_class = "Female", verbose = True)

# Train a model with SLIM
C0 = 1 / (2 * 100)
gamma = 0.1
SLIM_time = 10

SLIM_model = ScoringSystem(dataset)
SLIM(SLIM_model, C0, gamma, SLIM_time, random_seed, verbosity = False)

init_train_accuracy, init_test_accuracy = SLIM_model.training_accuracy(), SLIM_model.test_accuracy()
print("[INITIAL MODEL] train_accuracy =", init_train_accuracy, "test_accuracy =", init_test_accuracy)
SLIM_model.print()

SLIM_model.make_fairness_metrics(False)
print("[INITIAL MODEL] Initial statistical parity =", SLIM_model.SP)

# Optimize for fairness with our method, while staying in a 5%-Rashomon set and with a sparsity of at most 15 coefficients in the scoring system
perc_epsilon = 0.05
max_coeffs = 15 
fair_time = 10
fair_model = ScoringSystem(dataset)
fair_model.lambdas = SLIM_model.lambdas

constant_classifier_accuracy = max(np.count_nonzero(dataset.y_train == 1), np.count_nonzero(dataset.y_train == -1)) / n_train
translate_loss = (1 - init_train_accuracy) + perc_epsilon * (init_train_accuracy - constant_classifier_accuracy)
train_fair_Rashomon(fair_model, translate_loss, gamma, max_non_zeros = max_coeffs, time = fair_time, verbosity = False, random_key = random_seed, metric = "SP", reverse = False)

fair_train_accuracy, fair_test_accuracy = fair_model.training_accuracy(), fair_model.test_accuracy()
print("constant_classifier_accuracy =", constant_classifier_accuracy)
print("[FAIRER MODEL] new_train_accuracy =", fair_train_accuracy, "new_test_accuracy =", fair_test_accuracy)

fair_model.make_fairness_metrics(False)
fair_model.print()

print("[FAIRER MODEL] New statistical parity =", fair_model.SP)
assert(fair_model.SP <= SLIM_model.SP) #Statistical parity must be lower or equal
assert(n_features - fair_model.lambdas.count(0) <= max_coeffs) #The max features constraint must be respected
assert(1 - fair_train_accuracy <= translate_loss)
```

Expected output:

```bash
[INITIAL MODEL] train_accuracy = 0.98 test_accuracy = 0.7428298573673023


Scoring system: [0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0, -2.0, -2.0, 0.0, -5.0, 0.0, -10.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 10.0, 5.0, 1.0]

Score =  - 2 x ProfVocOrAS - 5 x JobService - 2 x JobSkilledSpecialty - 2 x JobAgriculture - 5 x DivorcedOrSeparated - 10 x NeverMarried + 2 x WorkHrsPerWeek_40_to_50 - 2 x OtherRace + 10 x AnyCapitalGains + 5 x AnyCapitalLoss + 1
Class = sign(Score)

[INITIAL MODEL] Initial statistical parity = 0.08683853459972862

constant_classifier_accuracy = 0.76

[FAIRER MODEL] new_train_accuracy = 0.97 new_test_accuracy = 0.720556976063584

Scoring system: [50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, -5.0, -2.0, -2.0, 1.0, -5.0, 1.0, -50.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -30.0, 0.0, 0.0, 50.0, 5.0, -50.0]


Score = 50 x Age_leq_21 - 2 x ProfVocOrAS + 20 x JobAdministrative - 5 x JobService - 2 x JobSkilledSpecialty - 2 x JobAgriculture + Married - 5 x DivorcedOrSeparated + Widowed - 50 x NeverMarried + 2 x WorkHrsPerWeek_40_to_50 - 30 x OtherRace + 50 x AnyCapitalGains + 5 x AnyCapitalLoss - 50
Class = sign(Score)

[FAIRER MODEL] New statistical parity = 0.01130710085933967
```

## Detailed Files Description
This repository is organized as follows:

* The `data` folder contains the three datasets we considered in our experiments (COMPAS, default credit and adult)
- The `figures` folder contains all the figures that have been generated for the paper (for scoring systems and decision diagrams).
- The `ssystems` folder contains all the code for scoring systems.
- The `ddiagrams` folder contains all the code for decision diagrams.

Within `ssystems`:

* The `models` subfolder contains all the scoring systems we found accross our experiments (for every sparsity and $\varepsilon$ values).
- `training.py` contains the implementation of SLIM and of our code to find the fairest model in a Rashomon set of scoring systems, according to a specific fairness metric.
- `plot_.py`: there are several files of this type, they save the different figures we display in the paper.
- `main_scoring.py` contains the script that launches the experiments (SLIM + fair train in Rashomon set), for a given experience id.
- `minimal_example.py` provides the previous minimal example of how to use our different training tools.
- `scoring_system.py` contains the implementation of our scoring system class that we use in different codes.
- `dataset.py` contains the implementation of our dataset class that we use in different codes.
- `harmonization_ssystems.py` implements solution sharing. For example, if we have a good fairness value for an epsilon of 0.01, this value can also be used for an epsilon of 0.1 because it is a relaxation of the problem. 

Within `ddiagrams`:

* `config.py` contains the configuration for decision diagrams learning.
- `dataset.py` which defines dataset structure for decision diagrams.
- `harmonization_ddiagrams.py` which implements solution sharing accross experiments (as for the scoring systems).
- `optimizer.py` which defines the MILP formulation for decision diagrams.
- `plot_.py`: there are several files of this type, they save the different figures we report in the paper.
- `solution.py` which defines the structure for solutions.
- `topology.py` which defines the structure for topology of decision diagrams
- `visualizer.py` which defines the structure for visualization of decision diagrams.


### How to launch experiments
The two important files to run our experiments are `main_scoring.py` (launch scoring systems experiments) and `main_ddiagrams` (launch decision diagrams experiments). They can be run with:

`python3.10 main_scoring.py --expe_id=xx` (for scoring systems) or `python3.10 main_ddiagrams.py --expe_id=xx` (for decision diagrams)

Where `xx` is the expe_id, given the desired combination of parameters the code should use to train ($\varepsilon$, $\alpha$, random seed). 
The associated different variables can also be set manually in the corresponding experiments' script.

As we did, all the experiments can be launched on a computing platform using Slurm through:

`sbatch job_ssystems.sh` (for scoring systems) or `sbatch job_ddiagrams.sh` (for decision diagrams)
