from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.discrete.discrete_model import GeneralizedPoisson
from statsmodels.tsa.ar_model import AutoReg
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from statsmodels.tools.tools import add_constant


class PoissonTimeSeries(BaseEstimator, RegressorMixin):

    def __init__(self):
        self.poisson_fit = None
        self.autoreg_fit = None

    def _fit_poisson(self, X, y):
        mod = GeneralizedPoisson(endog=y, exog=X)
        return mod.fit()

    def _fit_autoreg(self, endog, lags=1):
        mod = AutoReg(endog, lags)
        return mod.fit()

    def fit(self, X, y, **kwargs):
        X = add_constant(X)
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        self.poisson_fit = self._fit_poisson(X, y)
        poisson_resid = self.poisson_fit.resid
        self.autoreg_fit = self._fit_autoreg(poisson_resid)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = add_constant(X)
        X = check_array(X)
        poisson_pred = self.poisson_fit.predict(X)
        autoreg_pred = self.autoreg_fit.forecast(len(X))
        return poisson_pred + autoreg_pred

if __name__ == '__main__':
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    mod = PoissonTimeSeries()
    fit = mod.fit(x_train, y_train)
    mod.predict(x_test)
