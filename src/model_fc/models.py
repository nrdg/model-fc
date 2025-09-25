import numpy as np
from nilearn.connectome import ConnectivityMeasure
from pyuoi.linear_model import UoI_Lasso
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNetCV, LassoCV, LassoLarsIC
from sklearn.linear_model._base import _preprocess_data, _rescale_data
from sklearn.metrics import r2_score
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_is_fitted,
    check_X_y,
)


def run_model(train_ts, test_ts, n_rois, model, **kwargs):
    """Calculate a model based functional connectivity matrix.


    train_ts: training timeseries
    test_ts: testing timeseries
    n_rois: number of rois in parcellation
    model: model object


    """
    assert train_ts.shape[1] == n_rois == test_ts.shape[1]
    fc_mat = np.zeros((n_rois, n_rois))

    inner_rsq_dict = {"train": [], "test": []}

    for target_idx in range(train_ts.shape[1]):
        y_train = np.array(train_ts[:, target_idx])
        X_train = np.delete(train_ts, target_idx, axis=1)

        y_test = np.array(test_ts[:, target_idx])
        X_test = np.delete(test_ts, target_idx, axis=1)

        model.fit(X=X_train, y=y_train)

        fc_mat[target_idx, :] = np.insert(model.coef_, target_idx, 0)
        test_rsq, train_rsq = eval_metrics(X_train, y_train, X_test, y_test, model)

        inner_rsq_dict["test"].append(test_rsq)
        inner_rsq_dict["train"].append(train_rsq)

    return (fc_mat, inner_rsq_dict, model)


def eval_metrics(X_train, y_train, X_test, y_test, model):
    """Calculates R2 scores for FC models."""

    test_rsq = r2_score(y_test, model.predict(X_test))
    train_rsq = r2_score(y_train, model.predict(X_train))

    return (test_rsq, train_rsq)


def init_model(
    model_str, max_iter, random_state, stability_selection=16, selection_frac=0.7
):
    """Initialize model object for FC calculations."""
    if model_str == "uoi-lasso":
        uoi_lasso = UoI_Lasso(estimation_score="BIC")
        comm = MPI.COMM_WORLD

        uoi_lasso.selection_frac = selection_frac
        uoi_lasso.stability_selection = stability_selection
        uoi_lasso.copy_X = True
        uoi_lasso.estimation_target = None
        uoi_lasso.logger = None
        uoi_lasso.warm_start = False
        uoi_lasso.comm = comm
        uoi_lasso.random_state = 1
        uoi_lasso.n_lambdas = 100
        uoi_lasso.max_iter = max_iter
        uoi_lasso.random_state = random_state

        model = uoi_lasso

    elif model_str == "lasso-cv":
        lasso = LassoCV(
            fit_intercept=True,
            cv=5,
            n_jobs=-1,
            max_iter=max_iter,
            random_state=random_state,
        )

        model = lasso

    elif model_str == "lasso-bic":
        lasso = LassoLarsIC(criterion="bic", fit_intercept=True, max_iter=max_iter)

        model = lasso

    elif model_str == "enet":
        enet = ElasticNetCV(fit_intercept=True, cv=5, n_jobs=-1, max_iter=max_iter)
        model = enet

    elif model_str in ["correlation", "tangent"]:
        model = ConnectivityMeasure(kind=model_str)

    return model


class PearsonRegressor(BaseEstimator):
    """
    Parameters
    ----------
    """

    def __init__(self, fit_scale=True, copy_X=False, fit_intercept=False):
        self.fit_scale = fit_scale
        self.copy_X = copy_X
        self.fit_intercept = fit_intercept

    def _validate_input(self, X, y, sample_weight=None):
        """
        Helper function to validate the inputs
        """
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            copy=self.copy_X,
            sample_weight=sample_weight,
            check_input=True,
        )

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            outs = _rescale_data(X, y, sample_weight)
            X, y = outs[0], outs[1]

        return X, y, X_offset, y_offset, X_scale

    def fit(self, X, y, sample_weight=None):
        X, y, X_offset, y_offset, X_scale = self._validate_input(
            X, y, sample_weight=sample_weight
        )
        # Calculate the Pearson coefficient between y and every column of x
        corrmatrix = np.corrcoef(np.concatenate([X, y[:, None]], -1).T)
        self.coeff_ = corrmatrix[-1, :-1]
        # Calculate the linear combination of columns of x with these coefficients
        naive_pred_y = X @ self.coeff_
        # Calculathe the scale parameter to match the variance of y
        self.scale_ = naive_pred_y / y
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")
        # Multiply X by the coefficients
        print(X.shape)
        print(self.coeff_.shape)
        pred_naive = X @ self.coeff_
        # Multiply through by scale
        pred = pred_naive / self.scale_
        # Return
        return pred
