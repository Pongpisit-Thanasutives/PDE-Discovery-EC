import numpy as np
import statsmodels.api as sm
from collections import Counter
from pymcdm import weights as obj_w
from pymcdm.methods import TOPSIS, MABAC, COMET, SPOTIS
from pymcdm.methods.comet_tools import MethodExpert
from pymcdm import normalizations


def compromise_programming(
    best_subsets, dataset, weight="entropy_weights", normalization=None
):
    XX, yy = dataset

    ols_models = [sm.OLS(yy, XX[:, efi]).fit() for efi in best_subsets]
    rss = np.array([ols_model.ssr for ols_model in ols_models])
    bic = np.array([ols_model.bic for ols_model in ols_models])
    F = np.stack((rss, [len(ols_model.params) for ols_model in ols_models]), axis=1)

    obj_weights = getattr(obj_w, weight)(F, types=np.array([-1, -1]))
    F[:, 0:1] = -bic.reshape(-1, 1)
    if normalization is not None:
        F[:, 0:1] = getattr(normalizations, normalization)(F[:, 0:1])

    types = np.array([+1, -1])
    cvalues = COMET.make_cvalues(F)
    expert_function = MethodExpert(TOPSIS(), obj_weights, types)
    bounds = SPOTIS.make_bounds(F)

    method_names = ["TOPSIS", "MABAC", "COMET", "SPOTIS"]
    methods = [TOPSIS(), MABAC(), COMET(cvalues, expert_function), SPOTIS(bounds)]

    ranks = [method.rank(method(F, obj_weights, types)) for method in methods]
    ranks = Counter(np.argmin(ranks, axis=1)).most_common()
    balance_point = F[ranks[0][0]]

    return balance_point, ranks
