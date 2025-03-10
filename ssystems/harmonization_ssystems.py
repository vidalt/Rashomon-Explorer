'''
Script to "harmonize" scoring system models. 
It shares solution between different epsilons (if a solution works for an alpha on epsilon = 0.01
it also works for epsilon  = 0.2 for example, same if a solution works for an epsilon and 
alpha = 15, it also works for alpha = 16)
'''

#Parameters
epsilon_values = [0.01, 0.05, 0.1, 0.2]
n_trains = [500]
random_seeds = [100, 150, 200, 250, 300]
metrics = ["SP", "EO"]
datasets = ["credit", "adult", "compas"]
prefix_path = "ssystems/models/model-"

#Harmonization
for metric in metrics:
    for dataset in datasets:
        for n_train in n_trains:
            for random_seed in random_seeds:
                for reverse in [True, False]:
                    #First, for two successive value of epsilons, for every alpha, if the first epsilon's
                    #solution gives a better result than the second one, we give to the second one 
                    # the value of the first
                    for j in range (len(epsilon_values) - 1):
                        model_path = prefix_path + metric + "-" + dataset + "-" + str(n_train) + "-" + str(random_seed) + "-" + str(epsilon_values[j]) + "-" + str(reverse) + ".txt" 

                        with open(model_path, "r") as fichier:
                            content = fichier.read().split("\n")

                        new_path = prefix_path + metric + "-" + dataset + "-" + str(n_train) + "-" + str(random_seed) + "-" + str(epsilon_values[j + 1]) + "-" + str(reverse) + ".txt" 
                        with open(new_path, "r") as new_fichier:
                            new_content = new_fichier.read().split("\n")

                        for k in range (2, len(content)):
                            line1 = content[k].split(":")
                            line2 = new_content[k].split(":")
                            if len(line1) > 1:
                                if len(line2) == 1:
                                    new_content[k] = content[k]
                                else:
                                    sp_eps = float(line1[0])
                                    new_sp_eps = float(line2[0])
                                    if sp_eps < new_sp_eps:
                                        new_content[k] = content[k]

                        new_content = "\n".join(new_content)
                        with open(new_path, 'w') as f:
                            f.write(new_content)
                    
                    #Second, for every epsilon, for two successive values of alpha, if the first
                    #alpha's solution gives a better than the second one, we give to the second
                    #one the value of the first 
                    for j in range (len(epsilon_values)):
                        model_path = prefix_path + metric + "-" + dataset + "-" + str(n_train) + "-" + str(random_seed) + "-" + str(epsilon_values[j]) + "-" + str(reverse) + ".txt" 

                        with open(model_path, "r") as fichier:
                            content = fichier.read().split("\n")

                        for k in range (2, len(content) - 2):
                            line1 = content[k].split(":")
                            line2 = content[k + 1].split(":")
                            if len(line1) > 1:
                                if len(line2) == 1:
                                    content[k + 1] = content[k]
                                else:
                                    sp1 = float(line1[0])
                                    sp2 = float(line2[0])
                                    if sp1 < sp2:
                                        content[k + 1] = content[k]

                        content = "\n".join(content)
                        with open(model_path, 'w') as f:
                            f.write(content)
                    
                #Third, if we have a solution for an epsilon/alpha when maximizing or minimizing
                #but not for the other, we give the solution to the problem which doesn't have
                #solution
                for reverse in [True, False]:
                    other = (reverse == False)
                    for j in range (len(epsilon_values)):
                        model_path = prefix_path + metric + "-" + dataset + "-" + str(n_train) + "-" + str(random_seed) + "-" + str(epsilon_values[j]) + "-" + str(reverse) + ".txt" 

                        with open(model_path, "r") as fichier:
                            content = fichier.read().split("\n")

                        new_path = prefix_path + metric + "-" + dataset + "-" + str(n_train) + "-" + str(random_seed) + "-" + str(epsilon_values[j]) + "-" + str(other) + ".txt" 
                        with open(new_path, "r") as new_fichier:
                            new_content = new_fichier.read().split("\n")

                        for k in range (2, len(content)):
                            line1 = content[k].split(":")
                            line2 = new_content[k].split(":")
                            if len(line1) > 1:
                                if len(line2) == 1:
                                    if content[k][0] == "-":
                                        new_content[k] = content[k][1:]
                                    else:
                                        new_content[k] = "-" + content[k]

                        new_content = "\n".join(new_content)
                        with open(new_path, 'w') as f:
                            f.write(new_content)