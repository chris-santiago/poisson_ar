"""Module for Poisson TimeSeries Model"""
from typing import Optional, Union

from pandas import DataFrame, Series
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import PoissonRegressor
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from statsmodels.tsa.ar_model import AutoReg, AutoRegResults

Data = Union[DataFrame, Series, np.ndarray]


class PoissonAutoReg(BaseEstimator, RegressorMixin):
    """Estimator for Generalized Poisson model with autoregressive residuals."""
    def __init__(self, lags: int = 1):
        """Constructor"""
        self.X_: Optional[Data] = None
        self.y_: Optional[Data] = None
        self.lags = lags
        self.intercept_: Optional[float] = None
        self.coef_: Optional[Data] = None
        self.resid_: Optional[Data] = None
        self._poisson_fit: Optional[PoissonRegressor] = None
        self._autoreg_fit: Optional[AutoRegResults] = None

    def _fit_poisson(self) -> None:
        """Fit a Poisson model."""
        mod = PoissonRegressor()
        self._poisson_fit = mod.fit(self.X_, self.y_)
        self.intercept_ = mod.intercept_
        self.coef_ = mod.coef_
        self.resid_ = np.ravel(self.y_) - mod.predict(self.X_)

    def _fit_autoreg(self) -> None:
        """Fit an AR model."""
        mod = AutoReg(self.resid_, self.lags)
        self._autoreg_fit = mod.fit()

    def fit(self, X: Data, y: Data) -> "PoissonAutoReg":
        """Fit the estimator."""
        self.X_, self.y_ = check_X_y(X, y)
        self._fit_poisson()
        self._fit_autoreg()
        return self

    def predict(self, X: Data) -> np.ndarray:
        """Predict the response."""
        check_is_fitted(self)
        if self._poisson_fit is None:
            raise AttributeError('Model must be fit before predicting.')
        if self._autoreg_fit is None:
            raise AttributeError('Model must be fit before predicting.')
        x_test = check_array(X)
        poisson_pred = self._poisson_fit.predict(x_test)
        autoreg_pred = self._autoreg_fit.forecast(len(x_test))
        return poisson_pred + autoreg_pred
