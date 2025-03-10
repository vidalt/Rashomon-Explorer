import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

######## This script saves table figures of scoring systems, available in appendix ########

#Parameters for all the plots
perc_epsilon_values = [0.01, 0.05, 0.1, 0.2]
eps_str = ["1%", "5%", "10%", "20%"]
n_seeds = 5
random_seeds = [100 + 50 * i for i in range (n_seeds)]
metrics = ["SP", "EO"]
datasets = ["adult", "credit", "compas"]
real_reverse = {"adult" : True, "credit" : False, "compas" : False} #Let us know if we have to reverse the graphic or not

figures_sizes = (7.0,3.0)

maxmin = [["MAX", "MIN"], ["MIN", "MAX"], ["MIN", "MAX"]]
prefix_path = "ssystems/models/model-"

#Tables
for d, dataset in enumerate(datasets):
    for metric in metrics:
        fig = plt.figure(figsize = figures_sizes)
        ax = plt.gca()
        table = []
        MIN_C = 1000
        for r, reverse in enumerate([True, False]):
            if r == 1:
                table.append([maxmin[d][r] + " " + metric] + ["" for _ in range(len(table[0]) - 1)])
            for p, perc in enumerate(perc_epsilon_values):
                line = ["$\\epsilon = $" + eps_str[p]] #Means on all random seeds, for every maximal number of features
                
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
                coeff_values = [] 
                n_features = len(fairness_values[0])

                n = False
                min_coeff = 1
                for coeff in range (n_features):
                    fairness_values_coeff = []
                    nb = 0
                    for i in range (n_seeds):
                        val = fairness_values[i][coeff]
                        if val != "infeasible" and val != "no_sol" and val != "":
                            if reverse == real_reverse[dataset]:
                                fairness_values_coeff.append(-float(val))
                            else:
                                fairness_values_coeff.append(float(val))
                            nb +=1
                        elif val == "no_sol":
                            fairness_values_coeff.append("N/A")
                        elif val == "":
                            n = True

                    if nb == n_seeds: #We consider the mean, only if gurobi found a solution for each different seed
                        line.append(str(np.mean(fairness_values_coeff))[:5] + "\n" + "(" + str(np.std(fairness_values_coeff))[:5] + ")")
                        coeff_values.append("$\\alpha = $" + str(coeff + 1))
                    elif not n: #else, we fix it as "N/A"
                        min_coeff += 1
                        line.append("N/A")
                        coeff_values.append("$\\alpha = $" + str(coeff + 1))
                if len(table) == 0: #Rows names in the table
                    table.append([maxmin[d][r] + " " + metric] + coeff_values)
                table.append(line)
                if MIN_C > min_coeff:
                    MIN_C = min_coeff
        table = [[table[i][0]] + table[i][MIN_C:] for i in range (len(table))]
        final_table = ax.table(cellText=table, loc='center', edges = 'horizontal')
        final_table.set_fontsize(75)
        final_table.scale(1,1.5)
        ax.axis('off')
        prefix = "full_results_table_ssystems_" + dataset + "_" + metric + "_"
        plt.savefig('./figures/ssystems/result_tables/%s500_five_seeds_average.pdf' %prefix, bbox_inches = 'tight')
        plt.clf()