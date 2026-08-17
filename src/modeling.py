import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class RecipeFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["minutes"] = pd.to_numeric(X["minutes"], errors="coerce")
        X["n_steps"] = pd.to_numeric(X["n_steps"], errors="coerce")

        X["log_minutes"] = np.log1p(X["minutes"])
        X["steps_per_minute"] = X["n_steps"] / (X["minutes"] + 1)
        X["minutes_per_step"] = X["minutes"] / (X["n_steps"] + 1)

        return X