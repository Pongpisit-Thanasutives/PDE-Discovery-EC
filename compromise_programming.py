import numpy as np
import statsmodels.api as sm
from collections import Counter
from pymcdm import weights as obj_w
from pymcdm.methods import TOPSIS, MABAC, COMET, SPOTIS
from pymcdm.methods.comet_tools import MethodExpert
from pymcdm import normalizations


def compromise_programming(
    best_subsets,
    dataset,
    weight="entropy_weights",
    ssr_normalization=None,
    bic_normalization=None,
):
    XX, yy = dataset

    ssr = []
    bic = []
    complexity = []
    for efi in best_subsets:
        ols_result = sm.OLS(yy, XX[:, efi]).fit()
        ssr.append(ols_result.ssr)
        bic.append(ols_result.bic)
        complexity.append(len(ols_result.params))
    bic = np.array(bic)

    F = np.stack((ssr, complexity), axis=1)
    if ssr_normalization is not None:
        F[:, 0:1] = getattr(normalizations, ssr_normalization)(F[:, 0:1])
    obj_weights = getattr(obj_w, weight)(F, types=np.array([-1, -1]))

    F[:, 0:1] = -bic.reshape(-1, 1)
    if bic_normalization is not None:
        F[:, 0:1] = getattr(normalizations, bic_normalization)(F[:, 0:1])

    types = np.array([+1, -1])
    ranks = mcdm(F, obj_weights, types)
    ranks = Counter(np.argmin(ranks, axis=1)).most_common()
    balance_point = F[ranks[0][0]]

    return balance_point, ranks


def mcdm(F, obj_weights, types):
    cvalues = COMET.make_cvalues(F)
    expert_function = MethodExpert(TOPSIS(), obj_weights, types)
    bounds = SPOTIS.make_bounds(F)

    # method_names = ["TOPSIS", "MABAC", "COMET", "SPOTIS"]
    # print("method_names:", method_names)
    methods = [TOPSIS(), MABAC(), COMET(cvalues, expert_function), SPOTIS(bounds)]

    ranks = [method.rank(method(F, obj_weights, types)) for method in methods]
    return ranks
