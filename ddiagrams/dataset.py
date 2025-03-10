
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

from sklearn import preprocessing

class Dataset:
    """
    Loads a data set file into a format recognizable by other modules.

    Expects the path to a CSV file where each row is a data sample, and each column up to the last
    represents a feature. The last column is expected to be the target class.

    It is responsible for splitting data into training, validation and test sets, and provides attributes for each
    of these structures.

    It can also normalize feature values with sklearn's MinMaxScaler if `normalize_inputs=True`.
    
    Parameters
    ----------
    csv_file : str
        Path to CSV file.
    train_validation_test_split : list of float, default [1.0, 0.0, 0.0]
        List with three values representing the ratio of training, validation and test 
        data sets, respectively. The list must have exactly three values between 0.0 and 1.0.
        The split randomization is dependent on the `seed` parameter.
    normalize_inputs : bool, default True
        If set to True, normalizes data using sklearn's MinMaxScaler.
    seed : int or None, default None
        Random seed to be used for the train/validation/test split.
    verbose : bool, default False
        Flag to turn on verbose logging.

    Attributes
    ----------
    n : int
        Number of samples in the data set.
    p : int
        Number of features in the data set.
    features : list of int
        List of feature indexes.
    X : ndarray
        2D array of samples.
    Y : ndarray
        1D array of sample targets.
    classes : ndarray
        1D array of data set classes.
    train_n : int
        Number of training samples.
    train_X : ndarray
        2D array of training samples.
    train_Y : ndarray
        1D array of training sample targets.
    validation_n : int
        Number of validation samples.
    validation_X : ndarray
        2D array of validation samples.
    validation_Y : ndarray
        1D array of validation sample targets.
    test_n : int
        Number of test samples.
    test_X : ndarray
        2D array of test samples.
    test_Y : ndarray
        1D array of test sample targets.
    """

    def __init__(self, csv_file, train_size, normalize_inputs=True, seed=None, verbose=False, protected_class="", predictions_key=""):

        self.df = pd.read_csv(csv_file)
        self.verbose = verbose
        self.seed = seed
        self.train_n = train_size
        self.protected_class = protected_class
        self.predictions_key = predictions_key
        
        if normalize_inputs:
            V = self.df.values 
            min_max_scaler = preprocessing.MinMaxScaler()
            V_scaled = min_max_scaler.fit_transform(V)
            new_df = pd.DataFrame(V_scaled, columns = self.df.columns)
            class_column = self.df.columns[-1] # Assuming last column is class, which we do not want to normalize
            new_df[class_column] = self.df[class_column]
            self.df = new_df

        self.n = len(self.df)
        self.p = len(self.df.columns)-1
        self.features = [i for i in range(self.p)]
        
        self._setup_train_valid_test()
            
        if verbose:
            print("Observations:", self.n, "--- Training:", self.train_n,"Validation:",self.validation_n,"Test:", self.test_n)
            print("Features:", self.p)
            print("Classes:", len(self.classes))

    def _setup_train_valid_test(self):
        """Splits data into training, validation and test sets"""
        self.X = self.df.iloc[:, [i for i in range(self.p)]].values
        self.Y = self.df.iloc[:, self.p].values

        self.classes = np.unique(self.Y)
        sample_list = [i for i in range(self.n)]

        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(self.X, self.Y, train_size = self.train_n, random_state = self.seed)
        
        train_validation_size = 1

        
        self.validation_n = train_validation_size-self.train_n
        self.validation_X = self.df.iloc[sample_list[self.train_n:train_validation_size], [i for i in range(self.p)]].values
        self.validation_Y = self.df.iloc[sample_list[self.train_n:train_validation_size], self.p].values
        
        self.test_n = self.n - self.train_n
        self.Y1 = self.df.pop(self.predictions_key).values
        self.Xt, self.Xte, self.Yt, self.Yte = train_test_split(self.df, self.Y, train_size = self.train_n, random_state = self.seed)
    