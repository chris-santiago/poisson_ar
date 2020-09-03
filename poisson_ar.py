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
        self.X_ = None
        self.y_ = None
        self.lags_ = lags
        self.intercept_ = None
        self.coef_ = None
        self.alpha_ = None
        self._poisson_fit = None
        self._autoreg_fit = None

    def _fit_poisson(self, X, y):
        """Fit a Poisson model."""
        mod = GeneralizedPoisson(endog=y, exog=X)
        self._poisson_fit = mod.fit()
        params = self._poisson_fit.params
        self.intercept_ = params[0]
        self.alpha_ = params[-1]
        self.coef_ = params[1:-1]

    def _fit_autoreg(self, endog):
        """Fit an AR model."""
        mod = AutoReg(endog, self.lags_)
        self._autoreg_fit = mod.fit()

    def fit(self, X, y):
        """Fit the estimator."""
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        self._fit_poisson(add_constant(X), y)
        self._fit_autoreg(self._poisson_fit.resid)
        return self

    def predict(self, X):
        """Predict the response."""
        check_is_fitted(self)
        X = check_array(X)
        return self._poisson_fit.predict(add_constant(X)) + self._autoreg_fit.forecast(len(X))
