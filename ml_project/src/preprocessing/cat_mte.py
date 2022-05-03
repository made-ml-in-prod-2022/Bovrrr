import pandas as pd
import numpy as np

import category_encoders as ce

from sklearn.base import TransformerMixin
from typing import Union

import logging

logger = logging.getLogger("MTE")


class MTE(TransformerMixin):
    """Mean target encoding with shuffling"""

    def __init__(self, **params) -> None:
        """Using CatBoostEncoder from category_encoders"""
        self.encoder = ce.CatBoostEncoder(params)
        logger.debug("MTE instanced")

    def fit(
        self,
        X: Union[np.ndarray, pd.core.frame.DataFrame],
        y: Union[np.ndarray, pd.core.series.Series],
    ):
        assert X is not None and y is not None, f"{type(X), type(y)}"
        assert type(X) in [np.ndarray, pd.core.frame.DataFrame]
        assert type(y) in [np.ndarray, pd.core.series.Series]
        assert (
            X.shape[0] == y.shape[0]
        ), f"Size of X and y is different. {X.shape}, {y.shape}"
        assert (
            pd.Series(y).isna().sum() == 0
        ), f"y has nan values. {pd.Series(y).isna().sum()}"

        self.cols = []

        if isinstance(X, pd.core.frame.DataFrame):
            for feat in X.columns:
                if X[feat].nunique() < X.shape[0] / 10:
                    self.cols.append(feat)
        elif isinstance(X, np.ndarray):
            for feat in range(X.shape[1]):
                if X[:, feat].nunique() < X.shape[0] / 10:
                    self.cols.append(feat)

        self.encoder.set_params(cols=self.cols)

        logger.debug(f"cols: {self.cols}")

        logger.debug("MTE fit started")
        self.encoder.fit(X, y)
        logger.debug("MTE fit finished")
        return self

    def transform(
        self,
        X: Union[np.ndarray, pd.core.frame.DataFrame],
        y=Union[None, np.ndarray, pd.core.series.Series],
    ):
        assert X is not None, f"{type(X)}"

        X_ = self.encoder.transform(X, y)
        logger.debug("MTE transfromed")
        return X_
