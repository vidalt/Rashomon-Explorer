from sklearn.model_selection import train_test_split
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from random import *
import matplotlib.pyplot as plt

class Dataset:
    """
    Collection of structures that define a dataset.
    
    Parameters
    ----------
    X : panda frame
        the dataset, with every sample
    y : list of int
        class, for every sample in the dataset 
    train_size : int
        number of samples we use for training
    seed : int
        the random seed we use for gurobi and train/test split
    verbose : bool
        to activate the verbosity of our code
    protected_class : string
        the key for the group we want to protect
    ----------
     Attributes
    ----------
    n : int
        Number of samples in the data set.
    nb_features : int
        Number of features in the data set.
    X : ndarray
        2D array of samples.
    y : ndarray
        1D array of sample targets.
    train_n : int
        Number of training samples.
    X_train : ndarray
        2D array of training samples.
    y_train : ndarray
        1D array of training sample targets.
    test_n : int
        Number of test samples.
    X_test : ndarray
        2D array of test samples.
    y_test : ndarray
        1D array of test sample targets.
    protected_class : string
        the key for the group we want to protect
    """
    def __init__(self, X, y, train_size, seed, verbose = False, protected_class = ""):
        self.protected_class = protected_class

        self.X = X
        self.y = y
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size = train_size, random_state = seed)
    
        self.train_n = len(self.y_train)
        self.test_n = len(self.y_test)

        self.nb_features = len(self.X.keys())
        self.n = len(self.y)

        if verbose:
            print("Observations:", self.n, "--- Training:", self.train_n, "Test:", self.test_n)
            print("Features:", self.nb_features)
            print("Classes:", "-1 and 1")
