import numpy as np
import matplotlib.pyplot as plt
import json

#Parameters for all the plots
perc_epsilon_values = [0.01, 0.05, 0.1, 0.2]
eps_str = ["1%", "5%", "10%", "20%"]
random_seeds = [100, 150, 200, 250, 300]
n_seeds = len(random_seeds)
metrics = ["SP"]
datasets = ["credit"]
real_reverse = {"adult" : True, "credit" : False, "compas" : False} #Let us know if we have to reverse the graphic or not
show = ['-o', '--o']

colors = ["goldenrod", "darkslategray"]
figures_sizes = (7.0,4.0)
prefix_path = "ddiagrams/models/model-ddiagrams-"

#Plots
for dataset in datasets:
    for k, metric in enumerate(metrics):
        fig = plt.figure(figsize = figures_sizes)
        ax = plt.gca()
        for reverse in [True, False]:
            plt.xlabel("$\\epsilon$ of Rashomon set")
            if reverse == real_reverse[dataset]:
                plt.ylabel("Maximal SP")
            else:
                plt.ylabel("Minimal SP")
            for l, coeff in enumerate([4, 7]):
                eps_fair = []
                eps_std = []
                percs = []
                for j, perc in enumerate(perc_epsilon_values):
                    fairness_values = [] #fairness values for every different random seed and every maximal number of used features
                    for random_seed in random_seeds:
                        result_path = prefix_path + metric + "-" + dataset + "-500-" + str(random_seed) + "-" + str(perc) + "-" + str(reverse) + ".json" 
                        seed_fairness_values = []
                        with open(result_path, 'r') as outfile:
                            file = json.load(outfile)
                            for i in range (1, 13):
                                fairness = file[str(i)][metric]
                                seed_fairness_values.append(fairness)

                        fairness_values.append(seed_fairness_values)
                    fairness_means = [] #Means on all random seeds, for every maximal number of features
                    fairness_std = [] #Same with standard deviations
                    coeff_values = [] 
                    n_features = len(fairness_values[0])

                    values = []
                    for i in range (n_seeds):
                        val = fairness_values[i][coeff - 1]
                        if val != "infeasible" and val != "no_sol" and val != "":
                            if reverse == real_reverse[dataset]:
                                values.append(-float(val))
                            else:
                                values.append(float(val))

                        if len(values) == n_seeds: #We consider the mean, only if gurobi found a solution for each different seed
                            fairness_means.append(np.mean(values))
                            fairness_std.append(np.std(values))
                            coeff_values.append(14 + 1)
                    if len(fairness_means) >= 1:
                        eps_fair.append(fairness_means[0])
                        eps_std.append(fairness_std[0])
                        percs.append(perc)
                    
                plt.plot(percs, eps_fair, show[l], c = colors[l], label = "$\\alpha$=" + str(coeff) , alpha = 0.4)
                plt.fill_between(percs, np.asarray(eps_fair) - np.asarray(eps_std), np.asarray(eps_fair) + np.asarray(eps_std), color = colors[l], alpha = 0.2)
                plt.xticks(percs)
                plt.legend()
                print(eps_fair, eps_std)
            
            plt.savefig('./figures/ddiagrams/Figure_3_4_' + str(reverse) + '.pdf', bbox_inches = 'tight')
            plt.clf()