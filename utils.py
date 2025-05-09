from decimal import Decimal
import itertools

import numpy as np
from sklearn import linear_model
from sklearn.inspection import permutation_importance

from kneed import KneeLocator
from kneefinder import KneeFinder

import shap
try:
    # No need to import it if sage_linear_importance would never be called.
    import sage
except ImportError:
    print("sage is not installed to the environment.")

def knee(x, y, S=0.95, interp_method='linear', degree=7, direction='decreasing'):
    if direction == 'decreasing':
        curve = 'convex'
    elif direction == 'increasing':
        curve = 'concave'

    if interp_method == 'polynomial':
        opt = KneeLocator(x, y, S=S, interp_method='polynomial', polynomial_degree=degree, 
                          curve=curve, direction=direction).knee
    elif interp_method == 'linear':
        opt = KneeLocator(x, y, S=S, curve=curve, direction=direction).knee
    else:
        raise Exception("Unknown interp_method...")

    if direction == 'decreasing':
        argopt = x[np.argmin(y)]
    elif direction == 'increasing':
        argopt = x[np.argmax(y)]

    if opt is not None:
        opt = min(opt, argopt)
    else:
        opt = argopt
    return opt

def knee_finder(y, decreasing=False):
    y = np.array(y)
    if decreasing:
        decreasing_indices = range(0, len(y))
    else:
        decreasing_indices = decreasing_values_indices(y)
    if len(decreasing_indices) == 2 and y[1] < y[0]:
        return 1
    kf = KneeFinder(decreasing_indices, y[decreasing_indices])
    return int(kf.find_knee()[0])

def colvec(arr):
    return arr.reshape(-1, 1)

def select_column(arr, idx):
    assert len(arr.shape) == 2
    return arr[:, idx:idx+1]

def decreasing_values_indices(arr):
    mini = max(arr)+1; out = []
    for i, e in enumerate(arr):
        if e < mini:
            mini = e
            out.append(i)
    return np.array(out)

def distribute_order(n_poly, n_vars):
    out = []
    for n in range(n_poly+1):
        distribution = []
        for event in itertools.product(range(n+1), repeat=n_vars):
            if sum(event) == n:
                distribution.append(event)
        out.extend(sorted(distribution, reverse=True))
    return out

def sci_format(n):
    sf = '%.2E' % Decimal(n)
    sf = sf.split('E')
    return float(sf[0]), int(sf[1])

def permutation_linear_importance(X_pre, y_pre, n_repeats=30, scale=False, full=False):
    lm = linear_model.LinearRegression(fit_intercept=False).fit(X_pre, y_pre)
    pi = permutation_importance(lm, X_pre, y_pre, n_repeats=30)
    if not full:
        pi = pi['importances_mean']
        if scale:
            pi = pi/pi.sum()
    else:
        pi = pi['importances'].T
    return pi

def shap_linear_importance(X_pre, y_pre, scale=True, full=False):
    explainer = shap.explainers.Linear(linear_model.LinearRegression(fit_intercept=False).fit(X_pre, y_pre),
                                       X_pre)
    feature_importance = explainer(X_pre).values
    if not full:
        feature_importance = abs(feature_importance).mean(axis=0)
        if scale:
            feature_importance = feature_importance/feature_importance.sum()
    return feature_importance

def sage_linear_importance(X_pre, y_pre):
    imputer = sage.MarginalImputer(linear_model.LinearRegression(fit_intercept=False).fit(X_pre, y_pre), X_pre)
    estimator = sage.PermutationEstimator(imputer, 'mse')
    return estimator(X_pre_top, y_pre)

def extract_unique_candidates(pareto_optimal_models):
    unique_candidates = frozenset()
    for i in range(len(pareto_optimal_models)):
        unique_candidates = unique_candidates.union(pareto_optimal_models[i][0])
    return sorted(unique_candidates)

