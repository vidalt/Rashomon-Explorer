from dataset import Dataset
from topology import Topology
from heuristic import Heuristic
from optimizer import Optimizer
from visualizer import Visualizer
import argparse
import json
import numpy as np
from solution import *

parser = argparse.ArgumentParser(description = 'Train a fair decision diagram')
parser.add_argument('--expe_id', type = int, default = 15)
args = parser.parse_args()

expe_id = args.expe_id
use_previous_training = True
data_name = "adult"
metric = "EO"
compute_canada = True
reverse = True #reverse is to change what group is favored

if data_name == "compas":
    dataset = "data/compas.csv"
    predictions_key = "Recidivate-Within-Two-Years"
    protected_class = "Race=African-American"
elif data_name == "credit":
    dataset = "data/default_credit.csv"
    predictions_key = "DEFAULT_PAYEMENT"
    protected_class = "SEX_Female"
elif data_name == "adult":
    dataset = "data/adult_processed.csv"
    predictions_key = "Over50K"
    protected_class = "Female"

skeleton = [1,2,3,3,3]
perc_for_epsilon = [0.01, 0.05, 0.1, 0.2] #percents we use for epsilons
random_seeds = [100, 150, 200, 250, 300] #5 different seeds we use for expes
nb_nodes = 12
alpha = 0.0001
train_size = 500

#Load the dataset

if not use_previous_training:
    for random_seed in random_seeds:
        data = Dataset(dataset, train_size, seed=random_seed, protected_class=protected_class, predictions_key = predictions_key)
        topology = Topology(skeleton, data)

        #Find a first solution with heuristic
        heuristic = Heuristic(data, topology, alpha=alpha)
        print('Heuristic', heuristic.solution.training_accuracy(), heuristic.solution.test_accuracy())

        #Train initial model
        optimized = Optimizer(data, topology, alpha=alpha, initial_solution=heuristic.solution)
        print('Optimized', optimized.solution.training_accuracy(), optimized.solution.test_accuracy())

        SP = optimized.solution.SP(reverse)
        EO = optimized.solution.EO(reverse)
        PE = optimized.solution.PE(reverse)

        solution = {}
        solution["0"] = {
                "training_accuracy" : optimized.solution.training_accuracy(),
                "test_accuracy" : optimized.solution.test_accuracy()       
        }
        for i in range (1, optimized.solution.nb_active_nodes):
            solution[str(i)] = { 
                "status" : "no_known_sol",
                "SP": 2,
                "EO": 2,
                "PE": 2,
                "y" : {}, 
                "w_t" : {}, 
                "w_p" : {}, 
                "w_n" : {},
                "z" : {},
                "lambda_var" : {}
            }

        for i in range (optimized.solution.nb_active_nodes, nb_nodes + 1):
            solution[str(i)] = { 
                "status" : optimized.solution.status,
                "SP": SP,
                "EO": EO,
                "PE": PE,
                "y" : optimized.solution.y, 
                "w_t" : optimized.solution.w_t, 
                "w_p" : optimized.solution.w_p, 
                "w_n" : optimized.solution.w_n,
                "z" : optimized.solution.z,
                "lambda_var" : optimized.solution.lambda_var
            }
        for met in ["SP", "EO", "PE"]:
            for perc in perc_for_epsilon:
                model_name = "ddiagrams/models/model-ddiagrams-" + met + "-" + data_name + "-" + str(train_size) + "-" + str(random_seed) + "-" + str(perc) + "-" + str(reverse) + ".json"
                with open(model_name, "w") as outfile: 
                    json.dump(solution, outfile)
else:
    parameters = [] #parameters list for expe_id
    nb_nodes_values = [i for i in range (1, nb_nodes + 1)]

    for perc in perc_for_epsilon:
        for nb_max_nodes in nb_nodes_values:
            for random_seed in random_seeds:
                parameters.append([perc, nb_max_nodes, random_seed])
    print("parameters: " + str(len(parameters)))
    for expe_id in range (241, 119, -1):
        print(expe_id)

        expe_id = expe_id%(len(parameters))
        
        expe_parameters = parameters[expe_id]
        perc_epsilon, max_nodes, random_seed = expe_parameters[0], expe_parameters[1], expe_parameters[2]

        print(len(parameters))
        data = Dataset(dataset, train_size, seed=random_seed, protected_class=protected_class, predictions_key=predictions_key)
        topology = Topology(skeleton, data)
        constant_classifier_accuracy = max(np.count_nonzero(data.train_Y == 1), np.count_nonzero(data.train_Y == 0)) / train_size

        model_paths = "ddiagrams/models/model-ddiagrams-" + "SP" + "-" + data_name + "-" + str(train_size) + "-" + str(random_seed) + "-" + str(perc_epsilon) + "-" + str(reverse) + ".json" 

        solution = Solution(data, topology)
        with open(model_paths, "r") as outfile:
            file = json.load(outfile)
            train_accuracy = file["0"]["training_accuracy"]
            sol = file[str(max_nodes)]
            status = sol["status"]
            if status != "no_known_sol" and status != "infeasible":
                solution.y = sol["y"]
                solution.w_t = sol["w_t"]
                solution.w_p = sol["w_p"]
                solution.w_n = sol["w_n"]
                solution.z = sol["z"]
                solution.lambda_var = sol["lambda_var"]
            else:
                solution = None

        translate_loss = (1 - train_accuracy) + perc_epsilon * (train_accuracy - constant_classifier_accuracy)
        model_path = "ddiagrams/models/model-ddiagrams-" + metric + "-" + data_name + "-" + str(train_size) + "-" + str(random_seed) + "-" + str(perc_epsilon) + "-" + str(reverse) + ".json"
        if status != "infeasible":
            use_initial_solution = (status != "no_known_sol")
            #Train fair model in Rashomon set
            fair = Optimizer(data, topology, metric = metric, reverse = reverse, alpha=alpha, initial_solution=None, test=solution, fair=True, use_previous=use_initial_solution, max_nodes=max_nodes, translate_loss=translate_loss)

            with open(model_path, 'r') as outfile:
                file = json.load(outfile)
            change = False
            if fair.solution.status != "no_known_sol" and fair.solution.status != "infeasible":
                fairness = 0
                if metric == "SP":
                    fairness = fair.solution.SP(reverse)
                elif metric == "EO":
                    fairness = fair.solution.EO(reverse)
                elif metric == "PE":
                    fairness = fair.solution.PE(reverse)

                for c in range (max_nodes, nb_nodes):
                    if file[str(c)]["status"] != "infeasible" and file[str(c)]["status"] != "no_known_sol":
                        old_f = file[str(c)][metric]
                        
                        # m = {"SP" : file[str(c)]["SP"], "EO" : file[str(c)]["EO"], "PE" : file[str(c)]["PE"]}
                        if fairness < old_f:
                            # m[metric] = fairness
                            file[str(c)] = { 
                                "status" : fair.solution.status,
                                # "SP" : m["SP"],
                                # "EO" : m["EO"],
                                # "PE" : m["PE"],
                                "y" : fair.solution.y, 
                                "w_t" : fair.solution.w_t, 
                                "w_p" : fair.solution.w_p, 
                                "w_n" : fair.solution.w_n,
                                "z" : fair.solution.z,
                                "lambda_var" : fair.solution.lambda_var
                            }
                            file[str(c)][metric] = fairness
                            change = True
                    else:
                        # m = {"SP" : file[str(c)]["SP"], "EO" : file[str(c)]["EO"], "PE" : file[str(c)]["PE"]}
                        # m[metric] = fairness
                        file[str(c)] = { 
                            "status" : fair.solution.status,
                            # "SP" : m["SP"],
                            # "EO" : m["EO"],
                            # "PE" : m["PE"],
                            "y" : fair.solution.y, 
                            "w_t" : fair.solution.w_t, 
                            "w_p" : fair.solution.w_p, 
                            "w_n" : fair.solution.w_n,
                            "z" : fair.solution.z,
                            "lambda_var" : fair.solution.lambda_var
                        }
                        file[str(c)][metric] = fairness
                        change = True
            else:
                if fair.solution.status == "infeasible":
                    for i in range (1, max_nodes + 1):
                        file[str(i)]["status"] = "infeasible"
                        change = True
            if change:
                print("The model has changed")
                with open(model_path, "w") as outfile: 
                    json.dump(file, outfile)