from dataset import *
class ScoringSystem:
    """
    Collection of structures that define a scoring system.
    
    Parameters
    ----------
    data : Dataset
        Dataset instance used to build the scoring system.
    Attributes
    ----------
    lambdas : list of int (can be float if we define the lambda set with floats)
        list of different scores of our model
    SP : float
        statistical parity of our model
    PE : float
        predictive equity of our model
    EO : float
        equal opportunity of our model
    train_predictions : list of int (-1 or 1)
        list of class predictions for every training sample
    lambda_set : array of int
        every possible scores for every feature
    """

    def __init__(self, data):
        self.data = data

        self.lambdas = []
        self.lambda_set = [[1, -1, 2, -2, 5, -5, 10, -10, 20, -20, 30, -30, 50, -50] for _ in range(self.data.nb_features)]

        self.SP = None
        self.PE = None
        self.EO = None

        self.train_predictions = None   
        self.test_predictions = None     

    def training_accuracy(self):
        """Calculates and returns the train accuracy."""
        if self.train_predictions == None:
            self.make_predictions()
        n_train = len(self.data.X_train)
        return np.count_nonzero(self.train_predictions == self.data.y_train) / n_train
    
    def test_accuracy(self):
        """Calculates and returns the test accuracy."""
        if self.test_predictions == None:
            self.make_test_predictions()
        n_test = len(self.data.X_test)
        return np.count_nonzero(self.test_predictions == self.data.y_test) / n_test
    
    def make_predictions(self):
        """Calculates the predictions of a given scoring model, for training dataset."""
        train_predictions = []
        keys = self.data.X_train.keys()
        n_train = len(self.data.X_train)
        for i in range (n_train):
            sample_score = 0
            pred = 1
            for j, key in enumerate(keys):
                sample_score += self.lambdas[j] * self.data.X_train[key][self.data.X_train.index[i]]
            if sample_score <= 0:
                pred = -1
            train_predictions.append(pred)
        self.train_predictions = train_predictions
        return
    
    def make_test_predictions(self):
        """Calculates the predictions of a given scoring model, for training dataset."""
        test_predictions = []
        keys = self.data.X_test.keys()
        n_test = len(self.data.X_test)
        for i in range (n_test):
            sample_score = 0
            pred = 1
            for j, key in enumerate(keys):
                sample_score += self.lambdas[j] * self.data.X_test[key][self.data.X_test.index[i]]
            if sample_score <= 0:
                pred = -1
            test_predictions.append(pred)
        self.test_predictions = test_predictions
        return
    
    def make_SP(self, reverse):
        """Calculates and returns the statistical parity of a given scoring model, for training dataset."""
        if self.SP != None:
            return self.SP
        if self.train_predictions == None:
            self.make_predictions()
        s_non_class = 0
        s_class = 0
        nb_class = 0
        protected_class = self.data.protected_class
        for i, pred in enumerate(self.train_predictions):
            s_non_class += (pred == 1 and self.data.X_train[protected_class][self.data.X_train.index[i]] == 0)
            s_class += (pred == 1 and self.data.X_train[protected_class][self.data.X_train.index[i]] == 1)
            nb_class += (self.data.X_train[protected_class][self.data.X_train.index[i]] == 1)
        N = len(self.data.y_train)
        delta = (s_non_class / (N - nb_class)) - (s_class / nb_class)
        if reverse:
            delta = -delta
        self.SP = delta
        return delta
    
    def make_PE(self, reverse):
        """Calculates and returns the predictive equity of a given scoring model, for training dataset."""
        if self.PE != None:
            return self.PE
        if self.train_predictions == None:
            self.make_predictions()
        s = 0
        s1 = 0
        nb = 0
        nb1 = 0
        protected_class = self.data.protected_class
        for i, pred in enumerate(self.train_predictions):
            s += (pred == 1  and self.data.X_train[protected_class][self.data.X_train.index[i]] == 0 and self.data.y_train[i] == -1)
            s1 += (pred == 1 and self.data.X_train[protected_class][self.data.X_train.index[i]] == 1 and self.data.y_train[i] == -1)
            nb += (self.data.X_train[protected_class][self.data.X_train.index[i]] == 0 and self.data.y_train[i] == -1)
            nb1 += (self.data.X_train[protected_class][self.data.X_train.index[i]] == 1 and self.data.y_train[i] == -1)
        delta = s / nb - s1 / nb1
        if reverse:
            delta = -delta
        self.PE = delta
        return delta

    def make_EO(self, reverse):
        """Calculates and returns the equal opportunity of a given scoring model, for training dataset."""
        if self.EO != None:
            return self.EO
        if self.train_predictions == None:
            self.make_predictions()
        s = 0
        s1 = 0
        nb = 0
        nb1 = 0
        protected_class = self.data.protected_class
        for i, pred in enumerate(self.train_predictions):
            s += (pred == 1 and self.data.X_train[protected_class][self.data.X_train.index[i]] == 0 and self.data.y_train[i] == 1)
            s1 += (pred == 1 and self.data.X_train[protected_class][self.data.X_train.index[i]] == 1 and self.data.y_train[i] == 1)
            nb += (self.data.X_train[protected_class][self.data.X_train.index[i]] == 0 and self.data.y_train[i] == 1)
            nb1 += (self.data.X_train[protected_class][self.data.X_train.index[i]] == 1 and self.data.y_train[i] == 1)
        delta = s / nb - s1 / nb1
        if reverse:
            delta = -delta
        self.EO = delta
        return delta
        
    def make_fairness_metrics(self, reverse):
        '''Calculates every fairness metric'''
        self.make_SP(reverse)
        self.make_PE(reverse)
        self.make_EO(reverse)

    def protected_samples(self):
        '''Calculates every index in training set we need to calculate SP, PE and EO'''
        indexes = []
        comp_indexes = []
        indexes_PE = []
        comp_indexes_PE = []
        indexes_EO = []
        comp_indexes_EO = []
        protected_class = self.data.protected_class
        for i, index in enumerate(self.data.X_train.index):
            if self.data.X_train[protected_class][index] == 1:
                indexes.append(i)
                if self.data.y_train[i] == -1:
                    indexes_PE.append(i)
                else:
                    indexes_EO.append(i)
            else:
                comp_indexes.append(i)
                if self.data.y_train[i] == -1:
                    comp_indexes_PE.append(i)
                else:
                    comp_indexes_EO.append(i)
        return indexes, comp_indexes, indexes_PE, comp_indexes_PE, indexes_EO, comp_indexes_EO
    
    def print(self):
        '''Displays the scoring system'''
        keys = self.data.X_train.keys()
        print("\n")
        print("Scoring system: ", end = "")
        print(self.lambdas)
        print("\n")
        has_before = False
        print("Score = ", end = "")
        for k, key in enumerate(keys):
            l = self.lambdas[k]
            if l != 0:
                print((" + " * (l > 0) + " - " * (l < 0)) * (has_before or l < 0) + str(abs(int(l))) * (abs(l) != 1 or key == "intercept") + ( " x " * (abs(l) != 1) + key) * (key != "intercept"), end = "")
                has_before = True
        print("")
        print("Class = sign(Score)")
        print("\n")

    def write(self, path, metric):
        '''In the file, writes training accuracy, test accuracy and for each value of max_coeffs, the wanted metric value and the initial solution'''
        metrics = {"SP": self.SP, "PE": self.PE, "EO": self.EO}
        nb_model_coeffs = self.data.nb_features - self.lambdas.count(0) #number of non zero coefficients values
        with open(path, "w") as file:
            file.write("TrainingAccuracy: " + str(self.training_accuracy()) + "\n")
            file.write("TestAccuracy: " + str(self.test_accuracy()) + "\n")
            for _ in range (1, nb_model_coeffs): #we don't know if there is a solution for those values but we know for others
                file.write("no_sol\n")
            for _ in range(nb_model_coeffs, self.data.nb_features + 1):
                file.write(str(metrics[metric]) + ": ")
                for l, score in enumerate(self.lambdas):
                    file.write(str(int(score)) + " " * (l != len(self.lambdas) - 1))
                file.write("\n")
    
    def initialize(self, path, max_coeffs):
        '''Initializes the scoring system, with the file's model'''
        with open(path, "r") as f:
            file = f.read().split("\n")

            train_accuracy, test_accuracy = float(file[0].split(" ")[1]), float(file[1].split(" ")[1])

            scores = []
            stat = file[1 + max_coeffs]
            if stat != "no_sol" and stat != "infeasible":
                for k in stat.split(" ")[1:]:
                    scores.append(int(k))
            self.lambdas = scores
        return stat, train_accuracy, test_accuracy
    
    def modify(self, path, reverse, metric, status, max_coeffs):
        '''Modifies the model, only if gurobi found a better one'''
        with open(path, 'r') as f:
            content = f.read().split("\n")
        change = False
        if len(self.lambdas) != 0:
            self.make_fairness_metrics(reverse)
            metrics = {"SP": self.SP, "PE": self.PE, "EO": self.EO}
            s = str(metrics[metric]) + ": "
            for l, score in enumerate(self.lambdas):
                s += str(int(score)) + " " * (l != len(self.lambdas) - 1)
            for c in range (max_coeffs, self.data.nb_features + 1):
                if content[1 + c] != "infeasible" and content[1 + c] != "no_sol":
                    old_sp = float(content[1 + c].split(":")[0])
                    if metrics[metric] < old_sp:
                        content[1 + c] = s
                        change = True
                else:
                    content[1 + c] = s
                    change = True
        else:
            if status == 3:
                for c in range (1, max_coeffs + 1):
                    content[1 + c] = "infeasible"
                    change = True
        if change:
            print("The model has changed")
            content = "\n".join(content)
            with open(path, 'w') as f:
                f.write(content)

            