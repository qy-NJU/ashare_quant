from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseModel(ABC):
    """
    Abstract base class for models.
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def train(self, X, y):
        """
        Train the model from scratch.
        Args:
            X (pd.DataFrame or np.array): Features.
            y (pd.Series or np.array): Targets.
        """
        pass

    @abstractmethod
    def partial_train(self, X, y):
        """
        Incrementally train the model with new data.
        Args:
            X (pd.DataFrame or np.array): New features.
            y (pd.Series or np.array): New targets.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict signals.
        Args:
            X (pd.DataFrame or np.array): Features.
        Returns:
            np.array: Predictions.
        """
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
