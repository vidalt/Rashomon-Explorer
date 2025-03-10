import numpy as np
import matplotlib.pyplot as plt

#Parameters for all the plots
perc_epsilon_values = [0.2]
eps_str = ["20%"]
random_seeds = [100, 150, 200, 250, 300]
n_seeds = len(random_seeds)
metrics = ["SP", "EO"]
datasets = ["adult"]
real_reverse = {"adult" : True, "credit" : False, "compas" : False} #Let us know if we have to reverse the graphic or not

colors = ["red"]
figures_sizes = (7.0,4.0)
prefix_path = "ssystems/models/model-"

#Plots
for dataset in datasets:
    for k, metric in enumerate(metrics):
        fig = plt.figure(figsize = figures_sizes)
        plt.xlabel("Max. number of non-zero coefficients $\\alpha$")
        plt.ylabel(metric)
        ax = plt.gca()
        for reverse in [False]:
            for j, perc in enumerate(perc_epsilon_values):
                fairness_values = [] #fairness values for every different random seed and every maximal number of used features
                for random_seed in random_seeds:
                    result_path = prefix_path + metric + "-" + dataset + "-500-" + str(random_seed) + "-" + str(perc) + "-" + str(reverse) + ".txt" 
                    seed_fairness_values = []
                    with open(result_path, "r") as file:
                        total = file.read().split("\n")[2:]
                        for i, res in enumerate(total):
                            fairness = res.split(": ")[0]
                            seed_fairness_values.append(fairness)
                    fairness_values.append(seed_fairness_values)
                fairness_means = [] #Means on all random seeds, for every maximal number of features
                fairness_std = [] #Same with standard deviations
                coeff_values = [] 
                n_features = len(fairness_values[0])
                for coeff in range (n_features):
                    values = []
                    for i in range (n_seeds):
                        val = fairness_values[i][coeff]
                        if val != "infeasible" and val != "no_sol" and val != "":
                            if reverse == real_reverse[dataset]:
                                values.append(-float(val))
                            else:
                                values.append(float(val))

                    if len(values) == n_seeds: #We consider the mean, only if gurobi found a solution for each different seed
                        fairness_means.append(np.mean(values))
                        fairness_std.append(np.std(values))
                        coeff_values.append(coeff + 1)
                plt.plot(coeff_values, fairness_means, c = colors[j], label = eps_str[j], alpha = 0.4)
                plt.fill_between(coeff_values, np.asarray(fairness_means) - np.asarray(fairness_std), np.asarray(fairness_means) + np.asarray(fairness_std), color = colors[j], alpha = 0.2)
        plt.savefig('./figures/ssystems/Figure_2_' + metric + '.pdf', bbox_inches = 'tight')
        plt.clf()