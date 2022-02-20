"""
This file contains various methods of imputing numpy arrays.

Mainly used to fill missing values in the questionnaires.
"""
import re
from typing import Callable
import numpy as np
from sklearn.impute import KNNImputer

MISSING_VALUE = -1


class ImputationMethod():
    """
    Method for imputing missing data on binary data arrays (questionnaires in this case).
    Imputation methods do not change the data, they copy it (although this could be changed in
    the future if performance demands it.)

    RANDOM: Fills in a random value.
    k-NN: Fills in the value with the mean of the most common k neighbours, where k is an int.
    MEAN: Fills in the value with the dataset mean.
    """

    def __init__(self, method_name: str):
        """
        Initiates the imputation method. Additional arguments are
        given through the constructor and might be required for methods.
        Neighbours imputation f.e. needs to know how many neighbours to use.
        """
        self.method_name = method_name
        self.method = ImputationMethod._parse_imputation(method_name)

    def __str__(self) -> str:
        return "ImputationMethod(" + self.method_name + ")"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def _impute_random(data: np.ndarray):
        """
        Imputes missing values with a random value.
        """
        imputed_data = data.copy()
        imputed_data[imputed_data == -
                     1] = np.random.randint(0, 2, imputed_data[imputed_data == MISSING_VALUE].shape)
        return imputed_data

    @staticmethod
    def _impute_knn(data: np.ndarray, k: int):
        """
        Imputes missing values with the mean value of the k nearest neighbours. 
        Coinflip decides on 0.5.
        """
        print("Imputing via knn")
        imputer = KNNImputer(n_neighbors=k, missing_values=MISSING_VALUE)
        imputed_data = imputer.fit_transform(data)
        # removing the 0.5 values with random values
        imputed_data[imputed_data == 0.5] = np.random.randint(
            0, 2, imputed_data[imputed_data == 0.5].shape)
        return np.around(imputed_data)

    @staticmethod
    def _impute_mean(data: np.ndarray):
        """
        Imputes missing values with the mean value of the column.

        According to:
        https://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns
        """
        imputed_data = data.copy().astype(np.float32)
        imputed_data[imputed_data == MISSING_VALUE] = np.nan
        col_mean = np.nanmean(imputed_data, axis=0)
        inds = np.where(np.isnan(imputed_data))
        imputed_data[inds] = np.take(col_mean, inds[1])
        return imputed_data.round()

    @staticmethod
    def _parse_imputation(imputation_method_name: str) -> Callable:
        """
        Parses the imputation method from a string.
        """
        knn_regex = re.search(r"(\d+)-NN", imputation_method_name)
        if imputation_method_name.lower() == "random":
            return ImputationMethod._impute_random
        elif knn_regex is not None:
            k = knn_regex.group(1)
            return lambda x: ImputationMethod._impute_knn(x, int(k))
        elif imputation_method_name.lower() == "mean":
            return ImputationMethod._impute_mean
        else:
            raise ValueError(
                f"No valid imputation method was passed, received {imputation_method_name}")

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Imputes given data with the method used on construction.

        INPUT:
            data: np.ndarray
            Binary data (consisting of 0-1 values) in a nxm array to impute. Missing values are marked
            with a -1.
        OUTPUT:
            Imputed data of form nxw with w < n. No more values are allowed to be -1,
            these have been replaced with an imputation.
        """
        return self.method(data)
