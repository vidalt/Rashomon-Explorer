from training import *
from scoring_system import *
import pandas as pd

n_features = 37
random_seed = 40
n_train = 100

# Load data and subsample a training dataset with 100 examples
df = pd.read_csv("data/adult_processed.csv")
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
print("train_accuracy =", init_train_accuracy, "test_accuracy =", init_test_accuracy)
SLIM_model.print()

SLIM_model.make_fairness_metrics(False)
print("Initial statistical parity =", SLIM_model.SP, "\n")

#Improve fairness with our method, while staying in a 5%-Rashomon set
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
print("new_train_accuracy =", fair_train_accuracy, "new_test_accuracy =", fair_test_accuracy)

fair_model.make_fairness_metrics(False)
fair_model.print()

print("New statistical parity =", fair_model.SP)
assert(fair_model.SP <= SLIM_model.SP) #Statistical parity must be lower or equal
assert(n_features - fair_model.lambdas.count(0) <= max_coeffs) #The max features constraint must be respected
assert(1 - fair_train_accuracy <= translate_loss)