import json

epsilon_values = [0.01, 0.05, 0.1, 0.2]
n_trains = [500]
random_seeds = [100, 150, 200, 250, 300]
metrics = ["EO"]
datasets = ["adult"]
for metric in metrics:
    for dataset in datasets:
        for n_train in n_trains:
            for random_seed in random_seeds:
                for reverse in [True, False]:
                    for j in range (len(epsilon_values) - 1):
                        model_path = "ddiagrams/models/model-ddiagrams-" + metric + "-" + dataset + "-" + str(n_train) + "-" + str(random_seed) + "-" + str(epsilon_values[j]) + "-" + str(reverse) + ".json" 

                        with open(model_path, "r") as fichier:
                            content = json.load(fichier)

                        new_path = "ddiagrams/models/model-ddiagrams-" + metric + "-" + dataset + "-" + str(n_train) + "-" + str(random_seed) + "-" + str(epsilon_values[j + 1]) + "-" + str(reverse) + ".json" 
                        with open(new_path, "r") as new_fichier:
                            new_content = json.load(new_fichier)

                        for k in range (1, 13):
                            if content[str(k)]["status"] == "known_sol":
                                if content[str(k)]["status"] != "known_sol":
                                    new_content[str(k)] = content[str(k)]
                                else:
                                    sp_eps = content[str(k)][metric]
                                    new_sp_eps = new_content[str(k)][metric]
                                    if sp_eps < new_sp_eps:
                                        new_content[str(k)] = content[str(k)]

                        with open(new_path, 'w') as f:
                            json.dump(new_content, f)
                    
                    for j in range (len(epsilon_values)):
                        model_path = "ddiagrams/models/model-ddiagrams-" + metric + "-" + dataset + "-" + str(n_train) + "-" + str(random_seed) + "-" + str(epsilon_values[j]) + "-" + str(reverse) + ".json" 

                        with open(model_path, "r") as fichier:
                            content = json.load(fichier)

                        for k in range (1, 12):
                            if content[str(k)]["status"] == "known_sol":
                                if content[str(k + 1)]["status"] != "known_sol":
                                    content[str(k + 1)] = content[str(k)]
                                else:
                                    sp1 = content[str(k)][metric]
                                    sp2 = content[str(k + 1)][metric]
                                    if sp1 < sp2:
                                        content[str(k + 1)] = content[str(k)]

                        with open(model_path, 'w') as f:
                            json.dump(content, f)
                    

                # for reverse in [True, False]:
                #     other = (reverse == False)
                #     for j in range (len(epsilon_values)):
                #         model_path = "ddiagrams/models/model-ddiagrams-" + metric + "-" + dataset + "-" + str(n_train) + "-" + str(random_seed) + "-" + str(epsilon_values[j]) + "-" + str(reverse) + ".json" 

                #         with open(model_path, "r") as fichier:
                #             content = json.load(fichier)

                #         new_path = "ddiagrams/models/model-ddiagrams-" + metric + "-" + dataset + "-" + str(n_train) + "-" + str(random_seed) + "-" + str(epsilon_values[j]) + "-" + str(other) + ".json" 
                #         with open(new_path, "r") as new_fichier:
                #             new_content = json.load(new_fichier)

                #         for k in range (1, 13):
                #             line1 = content[str(k)]
                #             line2 = new_content[str(k)]
                #             if line1["status"] == "known_sol" and line2["status"] == "known_sol":
                #                 sp1 = content[str(k)][metric]
                #                 sp2 = -new_content[str(k)][metric]
                #                 if sp1 > sp2:
                #                     new_content[str(k)][metric] = content[str(k)][metric]

                #         with open(new_path, 'w') as f:
                #             json.dump(new_content, f)