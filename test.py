from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from poisson_ar import PoissonTimeSeries


def main():
    X, y = load_diabetes(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    mod = PoissonTimeSeries()
    fit = mod.fit(x_train, y_train)
    print(mod.score(x_train, y_train))
    print(mod.coef_)
    print(mod.intercept_)
    print(mod.alpha_)
    mod.predict(x_test)


if __name__ == '__main__':
    main()
