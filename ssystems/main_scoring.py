from training import *
from scoring_system import *
import argparse
import pandas as pd

if __name__ == "__main__":
    ############################## Parser extraction ##############################

    parser = argparse.ArgumentParser(description = 'Train a fair model')
    parser.add_argument('--expe_id', type = int, default = 0)
    args = parser.parse_args()

    expe_id = args.expe_id
    use_previous_training = True #if False, we initialize models with SLIM, else we train fair models with existing models
    data_name = "adult"
    metric = "SP"
    parallel_optimization = False
    reverse = False #reverse is to change what group is favored (if reverse = False, the algorithm
                    #will try to favour the protected_group to have prediction = 1)

    ############################## Global parameters for expes ##############################

    verbosity = True
    if parallel_optimization:
        verbosity = False
    n_train = 500
    time = 60
    perc_for_epsilon = [0.01, 0.05, 0.1, 0.2] #percents we use for epsilons
    random_seeds = [100, 150, 200, 250, 300] #5 different seeds we use for expes
    C0 = 1 / (2 * n_train)
    gamma = 0.1

    ############################## Datasets ##############################

    if data_name == "compas":
        csv_path = "data/compas.csv"
        predictions_key = "Recidivate-Within-Two-Years"
        protected_class = "Race=African-American"
        n_features = 28
    elif data_name == "credit":
        csv_path = "data/default_credit.csv"
        predictions_key = "DEFAULT_PAYEMENT"
        protected_class = "SEX_Female"
        n_features = 22
    elif data_name == "adult":
        csv_path = "data/adult_processed.csv"
        predictions_key = "Over50K"
        protected_class = "Female"
        n_features = 37

    X = pd.read_csv(csv_path)
    y = X.pop(predictions_key).values
    y -= (y == 0)
    X.insert(len(X.axes[1]), 'intercept', 1) #Intercept term
    prefix_path = "ssystems/models/model-"

    if not use_previous_training: #for Rashomon set's center
        ############################## SLIM training part ##############################

        for random_seed in random_seeds:
            dataset = Dataset(X, y, n_train, random_seed, protected_class = protected_class)
            init_scoring_system = ScoringSystem(dataset)

            SLIM(init_scoring_system, C0, gamma, time, random_seed, verbosity = verbosity, parallel_optimization = parallel_optimization)
            init_scoring_system.make_fairness_metrics(reverse)

            for met in ["SP", "EO"]:
                for perc in perc_for_epsilon:
                    model_name = prefix_path + met + "-" + data_name + "-" + str(n_train) + "-" + str(random_seed) + "-" + str(perc) + "-" + str(reverse) + ".txt"
                    init_scoring_system.write(model_name, met) #writes our model in a file
    else:
        ############################## Fairness part  ##############################

        parameters = [] #parameters list for expe_id
        coeffs_values = [i for i in range (1, n_features + 1)]

        for perc in perc_for_epsilon:
            for coeff in coeffs_values:
                for random_seed in random_seeds:
                    parameters.append([perc, coeff, random_seed])

        expe_parameters = parameters[expe_id]
        perc_epsilon, max_coeffs, random_seed = expe_parameters[0], expe_parameters[1], expe_parameters[2]
        
        dataset = Dataset(X, y, n_train, random_seed, protected_class = protected_class)
        fair_model = ScoringSystem(dataset)
        model_path = prefix_path + metric + "-" + data_name + "-" + str(n_train) + "-" + str(random_seed) + "-" + str(perc_epsilon) + "-" + str(reverse) + ".txt"
            
        stat, train_accuracy, test_accuracy = fair_model.initialize(model_path, max_coeffs) #initialize our model with existing model's values
        if stat != "infeasible": #we do not launch the training if we already proved it infeasible
            if verbosity and stat != "no_sol":
                fair_model.print()

            constant_classifier_accuracy = max(np.count_nonzero(dataset.y_train == 1), np.count_nonzero(dataset.y_train == -1)) / n_train
            translate_loss = (1 - train_accuracy) + perc_epsilon * (train_accuracy - constant_classifier_accuracy)
            status = train_fair_Rashomon(fair_model, translate_loss, gamma, max_non_zeros = max_coeffs, time = time, verbosity = verbosity, random_key = random_seed, metric = metric, reverse = reverse, parallel_optimization = parallel_optimization)
            
            fair_model.modify(model_path, reverse, metric, status, max_coeffs) #we modify the file if there is a change