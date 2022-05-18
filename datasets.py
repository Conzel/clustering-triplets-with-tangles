from enum import Enum
from data_generation import generate_gmm_data_fixed_means
from typing import Optional, Union
from tangles.data_types import Data
from questionnaire import Questionnaire
import numpy as np

from questionnaire import Questionnaire


class Dataset(Enum):
    GAUSS_SMALL = 1
    """Dataset with three clusters, 20 points each."""
    GAUSS_LARGE = 2
    """Dataset with three clusters, 200 points each."""
    GAUSS_MASSIVE = 3
    """NOT IMPLEMENTED"""

    @ staticmethod
    def get(en: int, seed: Optional[int]) -> Union[Data, tuple[np.ndarray, np.ndarray]]:
        """
        Returns the dataset described by the enum either as a Data object
        or as triplet-response combination (depends on dataset).
        """
        if en == Dataset.GAUSS_SMALL:
            means = np.array([[-6, 3], [-6, -3], [6, 3]])
            data = generate_gmm_data_fixed_means(
                20, means, std=1.0, seed=seed)
            return data
        elif en == Dataset.GAUSS_LARGE:
            means = np.array([[-6, 3], [-6, -3], [6, 3]])
            data = generate_gmm_data_fixed_means(
                200, means, std=0.7, seed=seed)
            return data
        else:
            raise ValueError(f"Dataset not supported (yet): {en}")


def toy_gauss_triplets(density: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = Dataset.get(Dataset.GAUSS_SMALL, seed=None)
    t, r = Questionnaire.from_metric(data.xs, density=density).to_bool_array()
    assert r is not None
    return t, r, data.ys
