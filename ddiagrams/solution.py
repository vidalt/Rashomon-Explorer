
import config as cfg

class Solution:
    """
    Collection of structures that define a built decision diagram.

    Instantiation of this class is intended for internal modules only.
    
    Parameters
    ----------
    data : Dataset
        Dataset instance used to build the decision diagram.
    topology : Topology
        Topology instance used to build the decision diagram.

    Attributes
    ----------
    used_nodes : Set
        Set of nodes used in the solution.
    node_hyperplane : dict of int: list of int 
        Splitting hyperplane of each used node.
        For univariate splits, the list will have all zeroes, except 
        for the index of the splitting feature.  
    node_intercept : dict of int: float
        Intercept of the splitting hyperplane of each used node.
    node_positive_arc : dict of int: int
        For each node, determines the child node its positive arc points to.
    node_negative_arc : dict of int: int
        For each node, determines the child node its negative arc points to.
    node_class : dict of int: int 
        Class assigned to each leaf node.
    mip_gap : float
        Optimality gap found by an Optimizer instance.
    obj_val : float
        Objective value found by an Optimizer instance.
    obj_lb : float
        Objective lower bound found by an Optimizer instance.
    bb_nodes : int
        Number of branch-and-cut nodes explored by an Optimizer instance.
    """

    def __init__(self, data, topology):
        self.data = data
        self.topology = topology
        self.status = "no_sol"
        self.nb_active_nodes = 0

        self.used_nodes = set()

        self.node_hyperplane = {}
        self.node_intercept = {}
        self.node_positive_arc = {}
        self.node_negative_arc = {}
        self.node_class = {}

        self.number_of_used_internal_nodes = None
        self.training_misclass = None
        self.validation_misclass = None
        self.test_misclass = None
        self.used_class_nodes = None

        self._training_samples_per_node = None
        self._validation_samples_per_node = None
        self._test_samples_per_node = None

        self.mip_gap = None
        self.obj_val = None
        self.obj_lb = None
        self.bb_nodes = None

        #Solution
        self.y =  {}

        # sample flow d.v.'s
        self.w_t = {}
        self.w_p = {}
        self.w_n = {}
        self.z = {}
        self.lambda_var = {}

    def training_accuracy(self):
        """Calculates and returns the training accuracy."""
        if self.training_misclass is None:
            self.training_misclass = self._calculate_misclass(self._calc_training_samples_per_node(),
                    self.data.train_Y)
        return 1 - self.training_misclass / self.data.train_n

    def validation_accuracy(self):
        """Calculates and returns the validation accuracy."""
        if self.validation_misclass is None:
            self.validation_misclass=self._calculate_misclass(self._calc_validation_samples_per_node(),
                    self.data.validation_Y)
        return 1 - self.validation_misclass / self.data.validation_n

    def test_accuracy(self):
        """Calculates and returns the test accuracy."""
        if self.test_misclass is None:
            self.test_misclass = self._calculate_misclass(self._calc_test_samples_per_node(),
                    self.data.test_Y)
        print(len(self.data.test_Y))
        return 1 - self.test_misclass / self.data.test_n

    def count_internal_nodes(self):
        """Calculates and returns the number of internal nodes."""
        if self.number_of_used_internal_nodes is None:
            used_internal_nodes = (self.used_nodes - {self.topology.root_node}
                    - set(self.node_class.keys()))
            self.number_of_used_internal_nodes = len(used_internal_nodes)
        return self.number_of_used_internal_nodes

    def objective_value(self, alpha, top_internal_nodes):
        """
        Calculates and returns the objective value.
        
        Parameters
        ----------
        alpha : float
            Alpha hyperparameter for controlling model complexity.
        top_internal_nodes : int
            Number of possible internal nodes in the diagram's topology.
        
        Returns
        -------
        objective_value : float
            Objective value for the solution instance.
        """
        if not self.topology.has_internal_nodes:
            return (1-self.training_accuracy())
        return ((1-self.training_accuracy()) + alpha * self.count_internal_nodes()
                / top_internal_nodes)

    def count_class_nodes(self):
        """Calculates and returns the number of used leaf nodes."""
        if self.used_class_nodes is None:
            self.used_class_nodes = len([u for u in self.node_class if u in self.used_nodes])
        return self.used_class_nodes

    def fragmentation_per_node(self):
        """Calculates and returns the proportion of training samples that reaches each used node."""
        frag = {}
        for l in range(self.topology.layers):
            for v in self.topology.nodes_per_layer[l]:
                if not v in self.used_nodes:
                    continue
                frag[v] = self._training_sample_proportion(v)
        return frag

    def print(self):
        """Prints a structured description of the decision diagram solution."""
        print("\n*** OCG layout ***")
        samples_per_node = self._calc_training_samples_per_node()
        for l in range(self.topology.layers):
            print("Layer",l)
            for v in self.topology.nodes_per_layer[l]:
                print("\tNode",v)
                if not v in self.used_nodes:
                    print("\t\tNot used")
                    continue
                if l==self.topology.layers-1:
                    print("\t\tClass:",self.topology.node_class[v])
                else:
                    print("\t\tSamples:",len(samples_per_node[v]))
                    print("\t\tSeparator:",self.node_hyperplane[v], self.node_intercept[v])
                    print("\t\tPositive arc to", self.node_positive_arc[v])
                    print("\t\tNegative arc to", self.node_negative_arc[v])
    
    def __samples_per_node(self, n, X, training):
        samples_per_node = {}
        samples_per_node[self.topology.root_node] = [i for i in range(n)]
        for l in range(1, self.topology.layers):
            for v in self.topology.nodes_per_layer[l]:
                samples_per_node[v] = []
        for l in range(self.topology.layers-1):
            for v in self.topology.nodes_per_layer[l]:
                for i in samples_per_node[v]:
                    delta = (sum([x*y for x,y in zip(self.node_hyperplane[v],X[i])])
                            - self.node_intercept[v])
                    if delta >= -cfg.solver_feas_tol:
                        samples_per_node[self.node_positive_arc[v]].append(i)
                    elif delta <= -cfg.epsilon+cfg.solver_feas_tol:
                        samples_per_node[self.node_negative_arc[v]].append(i)
                    else:
                        if training:
                            print("warning: training data point within forbidden epsilon range")
                            # we let this pass because Gurobi, sometimes, may return an "optimal"
                            # solution that does not respect the tolerance parameters
                            print("(look for \"max constraint violation\" Gurobi warning)")
                        else:
                            print("info: (non-training) data point within forbidden epsilon range")
                        if delta >= -cfg.epsilon/2:
                            samples_per_node[self.node_positive_arc[v]].append(i)
                        else:
                            samples_per_node[self.node_negative_arc[v]].append(i)
        return samples_per_node

    def _calculate_misclass(self, samples_per_node, Y):
        calculated_misclass = 0
        for v in self.topology.nodes_per_layer[self.topology.layers-1]:
            for i in samples_per_node[v]:
                if Y[i] != self.node_class[v]:
                    calculated_misclass += 1
        return calculated_misclass
    
    def SP(self, reverse):
        s_non_class = 0
        s_class = 0
        nb_class = 0
        protected_class = self.data.protected_class
        for v in self.topology.nodes_per_layer[self.topology.layers-1]:
            for i in self._calc_training_samples_per_node()[v]:
                s_non_class += (self.node_class[v] == 1 and self.data.Xt[protected_class][self.data.Xt.index[i]] == 0)
                s_class += (self.node_class[v] == 1 and self.data.Xt[protected_class][self.data.Xt.index[i]] == 1)
                nb_class += (self.data.Xt[protected_class][self.data.Xt.index[i]] == 1)

        N = len(self.data.train_Y)
        delta = (s_non_class / (N - nb_class)) - (s_class / nb_class)
        if reverse:
            delta = -delta
        return delta
    
    def PE(self, reverse):
        s = 0
        s1 = 0
        nb = 0
        nb1 = 0
        protected_class = self.data.protected_class
        for v in self.topology.nodes_per_layer[self.topology.layers-1]:
            for i in self._calc_training_samples_per_node()[v]:
                s += (self.node_class[v] == 1  and self.data.Xt[protected_class][self.data.Xt.index[i]] == 0 and self.data.Yt[i] == 0)
                s1 += (self.node_class[v] == 1 and self.data.Xt[protected_class][self.data.Xt.index[i]] == 1 and self.data.Yt[i] == 0)
                nb += (self.data.Xt[protected_class][self.data.Xt.index[i]] == 0 and self.data.Yt[i] == 0)
                nb1 += (self.data.Xt[protected_class][self.data.Xt.index[i]] == 1 and self.data.Yt[i] == 0)
        delta = s / nb - s1 / nb1
        if reverse:
            delta = -delta
        return delta

    def EO(self, reverse):
        s = 0
        s1 = 0
        nb = 0
        nb1 = 0
        protected_class = self.data.protected_class
        for v in self.topology.nodes_per_layer[self.topology.layers-1]:
            for i in self._calc_training_samples_per_node()[v]:
                s += (self.node_class[v] == 1 and self.data.Xt[protected_class][self.data.Xt.index[i]] == 0 and self.data.Yt[i] == 1)
                s1 += (self.node_class[v] == 1 and self.data.Xt[protected_class][self.data.Xt.index[i]] == 1 and self.data.Yt[i] == 1)
                nb += (self.data.Xt[protected_class][self.data.Xt.index[i]] == 0 and self.data.Yt[i] == 1)
                nb1 += (self.data.Xt[protected_class][self.data.Xt.index[i]] == 1 and self.data.Yt[i] == 1)
        delta = s / nb - s1 / nb1
        if reverse:
            delta = -delta
        return delta

    def _calc_training_samples_per_node(self):
        if self._training_samples_per_node is None:
            self._training_samples_per_node = self.__samples_per_node(self.data.train_n,
                    self.data.train_X, True)
        return self._training_samples_per_node

    def _calc_validation_samples_per_node(self):
        if self._validation_samples_per_node is None:
            self._validation_samples_per_node = self.__samples_per_node(self.data.validation_n,
                    self.data.validation_X, False)
        return self._validation_samples_per_node

    def _calc_test_samples_per_node(self):
        if self._test_samples_per_node is None:
            self._test_samples_per_node = self.__samples_per_node(self.data.test_n,
                    self.data.test_X, False)
        return self._test_samples_per_node

    def _training_sample_proportion(self, node):
        samples_per_node = self._calc_training_samples_per_node()
        return len(samples_per_node[node]) / self.data.train_n