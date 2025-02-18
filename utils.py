from decimal import Decimal
import shap
from sklearn import linear_model
# from kneed import KneeLocator

def colvec(arr):
    return arr.reshape(-1, 1)

def sci_format(n):
    sf = '%.2E' % Decimal(n)
    sf = sf.split('E')
    return float(sf[0]), int(sf[1])

def shap_linear_importance(X_pre, y_pre, scale=True):
    explainer = shap.explainers.Linear(linear_model.LinearRegression(fit_intercept=False).fit(X_pre, y_pre),
                                       X_pre)
    feature_importance = abs(explainer(X_pre).values).mean(axis=0)
    if scale:
        feature_importance = feature_importance/sum(feature_importance)
    return feature_importance

