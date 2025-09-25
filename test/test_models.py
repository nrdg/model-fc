import numpy as np
from model_fc.models import PearsonRegressor

# @parametrize_with_checks([PearsonRegressor()])
# def test_sklearn_compatible_estimator(estimator, check):
#     check(estimator)


def test_PearsonRegressor():
    X = np.random.randn(100, 20)
    y = np.random.randn(100)

    pr = PearsonRegressor()
    pr.fit(X, y)
    pr.predict(X)
