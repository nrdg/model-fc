import numpy as np
from mpi4py import MPI
from nilearn.connectome import ConnectivityMeasure
from pyuoi.linear_model import UoI_Lasso
from sklearn.linear_model import ElasticNetCV, LassoCV, LassoLarsIC
from sklearn.metrics import r2_score


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
