"""Module for Generalized Poisson TimeSeries Model"""
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from statsmodels.discrete.discrete_model import GeneralizedPoisson
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.ar_model import AutoReg


class PoissonTimeSeries(BaseEstimator, RegressorMixin):
    """Estimator for Generalized Poisson model with autoregressive residuals."""
    def __init__(self, lags=1):
        """Constructor"""
        self.lags = lags

    def _fit_poisson(self, X, y):
        """Fit a Poisson model."""
        mod = GeneralizedPoisson(endog=y, exog=X)
        self.poisson_fit_ = mod.fit()
        params = self.poisson_fit_.params
        self.intercept_ = params[0]
        self.alpha_ = params[-1]
        self.coef_ = params[1:-1]

    def _fit_autoreg(self, endog):
        """Fit an AR model."""
        mod = AutoReg(endog, self.lags)
        self.autoreg_fit_ = mod.fit()

    def fit(self, X, y):
        """Fit the estimator."""
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self._fit_poisson(add_constant(X), y)
        self._fit_autoreg(self.poisson_fit_.resid)
        return self

    def predict(self, X):
        """Predict the response."""
        check_is_fitted(self)
        X = check_array(X)
        return self.poisson_fit_.predict(add_constant(X)) + self.autoreg_fit_.forecast(len(X))
