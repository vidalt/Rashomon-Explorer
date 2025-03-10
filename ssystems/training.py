from scoring_system import *

def SLIM(init, C0, gamma, time, random_key, verbosity = 1, parallel_optimization = False):
    '''SLIM training, to obtain initial accurate model we then use as center of Rashomon set'''
    #Constants
    p = init.data.nb_features
    N = init.data.train_n
    keys = init.data.X_train.keys()
    minLambdas = [np.min(l) for l in init.lambda_set]
    maxLambdas = [np.max(l) for l in init.lambda_set]
    C1 = min(1 / N, C0) / (p * np.max(maxLambdas))

    model = gp.Model('SLIM')

    #Variables
    lambdas = [model.addVar(lb = minLambdas[j], ub = maxLambdas[j], vtype = GRB.INTEGER, name = "lambd" + str(j)) for j in range (p)]
    u = [[model.addVar(vtype = GRB.BINARY, name = "u" + str(omega) + "-" + str(j)) for omega in range (len(init.lambda_set[j]))] for j in range (p)]

    alphas = [model.addVar(vtype = GRB.BINARY, name = "alpha" + str(j)) for j in range (p)]
    betas = [model.addVar(lb = 0, ub = np.max(maxLambdas), vtype = GRB.INTEGER, name = "beta" + str(j)) for j in range (p)]
    z = [model.addVar(vtype = GRB.BINARY, name = "z" + str(i)) for i in range (N)]

    #Constraints
    for j in range (p):
        model.addConstr(maxLambdas[j] * alphas[j]  >= - lambdas[j])
        model.addConstr(maxLambdas[j] * alphas[j] >= lambdas[j])

        model.addConstr(betas[j] >= lambdas[j])
        model.addConstr(betas[j] >= -lambdas[j])

        Omega_j = len(init.lambda_set[j])
        model.addConstr(lambdas[j] == gp.quicksum(u[j][omega] * init.lambda_set[j][omega] for omega in range (Omega_j)))
        model.addConstr(gp.quicksum(U for U in u[j]) <= 1)

    BigM = [1000 for _ in range (N)] #TODO

    for i, ind in enumerate(init.data.X_train.index):
        model.addConstr(BigM[i] * z[i] >= (gamma - init.data.y_train[i] * gp.quicksum(init.data.X_train[keys[j]][ind] * lambdas[j] for j in range (p))))

    #Objective
    obj = gp.quicksum(z[i] for i in range(N)) + N * gp.quicksum(C0 * alphas[j] + C1 * betas[j] for j in range(p))
    model.setObjective(obj, GRB.MINIMIZE)

    #Solve the problem
    model.setParam('LogToConsole', verbosity) # 0 or 1
    model.Params.method = 3
    model.setParam('TimeLimit', time)
    if parallel_optimization:
        model.setParam('Threads', 16)
    else: 
        model.setParam('Threads', 8)
    model.setParam('Seed', random_key) # for reproducibility
    model.optimize()

    #Store variables to know where to start in fairness problem
    if model.SolCount != 0:
        scores = [lambdas[j].X for j in range (p)]
        init.lambdas = scores

def train_fair_Rashomon(init, translate_loss, gamma, max_non_zeros = 100, time = 30, verbosity = 1, random_key = 2, metric = "SP", reverse = False, parallel_optimization = False):
    '''Train a fair model in the Rashomon set from a given initial accurate model'''
    #Constants
    p = init.data.nb_features
    N = len(init.data.X_train)
    keys = init.data.X_train.keys()
    SP_index, n_SP_index, PE_index, n_PE_index, EO_index, n_EO_index = init.protected_samples()

    minLambdas = [np.min(init.lambda_set[i]) for i in range(p)]
    maxLambdas = [np.max(init.lambda_set[i]) for i in range(p)]

    #Declare the model
    model = gp.Model('FAIRSLIM')

    #Variables
    lambdas = [model.addVar(lb = minLambdas[j], ub = maxLambdas[j], vtype = GRB.INTEGER, name = "lambd" + str(j)) for j in range (p)]
    if len(init.lambdas) > 0:
        for j in range(p):
            lambdas[j].Start = init.lambdas[j]

    u = [[model.addVar(vtype = GRB.BINARY, name = "u" + str(omega) + "-" + str(j)) for omega in range (len(init.lambda_set[j]))] for j in range (p)]
    z = [model.addVar(vtype = GRB.BINARY, name = "z" + str(i)) for i in range (N)]

    m = [(init.data.y_train[i] == 1) for i in range (len(init.data.y_train))]
    o = [(init.data.y_train[i] == -1) for i in range (len(init.data.y_train))]

    #Constraints and objective
    model.addConstr((1 / N) * gp.quicksum(z[i] for i in range(N)) <= translate_loss)

    tot_non_zeros = 0
    for j in range (p):
        Omega_j = len(init.lambda_set[j])
        model.addConstr(lambdas[j] == gp.quicksum(u[j][omega] * init.lambda_set[j][omega] for omega in range (Omega_j)))
        model.addConstr(gp.quicksum(u[j][omega] for omega in range (Omega_j)) <= 1)
        tot_non_zeros += gp.quicksum(U for U in u[j])
    model.addConstr(tot_non_zeros <= max_non_zeros)

    BigM = [1000 for _ in range (N)] #TODO
    BigO = [1000 for _ in range (N)] #TODO

    for i, ind in enumerate(init.data.X_train.index):
        model.addConstr(BigM[i] * z[i] >= (gamma - init.data.y_train[i] * gp.quicksum(init.data.X_train[keys[j]][ind] * lambdas[j] for j in range (p))))
        model.addConstr(BigO[i] * (1 - z[i]) >= init.data.y_train[i] * gp.quicksum(init.data.X_train[keys[j]][ind] * lambdas[j] for j in range (p)))

    #The objective depends of the metric
    delta = 0
    if metric == "SP":
        delta = (len(SP_index) * gp.quicksum((1 - z[i]) * m[i] + o[i] * z[i] for i in n_SP_index) - (N - len(SP_index)) * gp.quicksum((1 - z[i]) * m[i] + o[i] * z[i] for i in SP_index))
    elif metric == "PE":
        delta = len(PE_index) * gp.quicksum((1 - z[i]) * m[i] + o[i] * z[i] for i in n_PE_index) - len(n_PE_index) * gp.quicksum((1 - z[i]) * m[i] + o[i] * z[i] for i in PE_index)
    elif metric == "EO":
        delta = len(EO_index) * gp.quicksum((1 - z[i]) * m[i] + o[i] * z[i] for i in n_EO_index) - len(n_EO_index) * gp.quicksum((1 - z[i]) * m[i] + o[i] * z[i] for i in EO_index)

    if reverse:
        model.setObjective(-delta, GRB.MINIMIZE)
    else:
        model.setObjective(delta, GRB.MINIMIZE)

    model.setParam('LogToConsole', verbosity) # 0 or 1
    model.Params.method = 3
    model.setParam('TimeLimit', time)
    if parallel_optimization:
        model.setParam('Threads', 16)
    else: 
        model.setParam('Threads', 8)
    model.setParam('Seed', random_key) # for reproducibility
    model.optimize()

    scores = []
    if model.SolCount != 0:
        scores = [lambdas[j].X for j in range (p)]
        init.lambdas = scores
    return model.Status