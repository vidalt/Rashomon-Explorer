import numpy as np
import matplotlib.pyplot as plt
import json

#Parameters for all the plots
perc_epsilon_values = [0.01, 0.05, 0.1, 0.2]
eps_str = ["1%", "5%", "10%", "20%"]
random_seeds = [100, 150, 200, 250, 300]
n_seeds = len(random_seeds)
metrics = ["SP", "EO"]
datasets = ["adult", "credit", "compas"]
real_reverse = {"adult" : True, "credit" : False, "compas" : False} #Let us know if we have to reverse the graphic or not

colors = ["blue", "orange", "green", "red", "purple"]
figures_sizes = (7.0,4.0)
prefix_path = "ddiagrams/models/model-ddiagrams-"

#Plots
for dataset in datasets:
    for k, metric in enumerate(metrics):
        fig = plt.figure(figsize = figures_sizes)
        plt.xlabel("Max. number of active nodes $\\alpha$")
        plt.ylabel(metric)
        ax = plt.gca()
        for reverse in [True, False]:
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

                for coeff in range (12):
                    values = []
                    for i in range (n_seeds):
                        val = fairness_values[i][coeff]
                        if float(val) != 2:
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
        prefix = "full_results_plot_ddiagrams_" + dataset + "_" + metric + "_"
        plt.savefig('./figures/ddiagrams/full_plots/%s500_five-seeds_average.pdf' %prefix, bbox_inches = 'tight')
        plt.clf()

#Legend
legend_fig = plt.figure("Legend plot")
legend_elements = []
for i, perc in enumerate(perc_epsilon_values):
    legend_elements.append(plt.Line2D([0], [0], marker=None, color=colors[i], lw = 1, label = "$\\epsilon = $" + eps_str[i]))
legend_fig.legend(handles=legend_elements, loc = 'center', ncol = len(eps_str))
legend_fig.savefig('./figures/ddiagrams/full_plots/full_results_plot_ddiagrams_legend.pdf', bbox_inches = 'tight')

