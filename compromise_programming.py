import numpy as np
import statsmodels.api as sm
from collections import Counter
from bayesian_model_evidence import log_evidence
# MCDM/MCDA
from pymcdm import weights as obj_w
from pymcdm.methods import TOPSIS, MABAC, COMET, SPOTIS
from pymcdm.methods.comet_tools import MethodExpert
from pymcdm import normalizations


def compromise_programming(
    best_subsets,
    dataset,
    weight="entropy_weights",
    criterion="bic",
    ssr_normalization=None,
    ic_normalization=None,
    ic_kwargs={},
    verbose=True
):
    assert criterion in {"bic", "bme"}
    XX, yy = dataset

    ssr = []
    ic = []
    complexity = []
    for efi in best_subsets:
        # SSR and Complexity
        ols_result = sm.OLS(yy, XX[:, efi]).fit()
        ssr.append(ols_result.ssr)
        complexity.append(len(ols_result.params))
        # Criterion: BIC or Bayesian Model Evidence (BME)
        if criterion == "bme":
            ic.append(-log_evidence(XX[:, efi], yy, **ic_kwargs))
        else:
            ic.append(ols_result.bic)
    ic = np.array(ic)

    F = np.stack((ssr, complexity), axis=1)
    n_alternatives, n_criteria = F.shape
    types = np.array([-1 for _ in range(n_criteria)]) # default to minimization
    if ssr_normalization is not None:
        if np.any(F[:, 0:1] < 0):
            F[:, 0:1] = -getattr(normalizations, ssr_normalization)(-F[:, 0:1])
        else:
            F[:, 0:1] = getattr(normalizations, ssr_normalization)(F[:, 0:1])
    obj_weights = getattr(obj_w, weight)(F, types=np.array(types))

    filtered_F = F.copy()
    filtered_F[:, 0:1] = ic.reshape(-1, 1)
    if ic_normalization is not None:
        if np.any(filtered_F[:, 0:1] < 0):
            filtered_F[:, 0:1] = -getattr(normalizations, ic_normalization)(-filtered_F[:, 0:1])
        else:
            filtered_F[:, 0:1] = getattr(normalizations, ic_normalization)(filtered_F[:, 0:1])

    decision_seq = []
    preference_seq = []
    while len(filtered_F) > 2:
        ranks, prefs = mcdm(filtered_F, obj_weights, types)
        balance_point, most_common = ranks2decision(ranks)
        if verbose:
            print(filtered_F, most_common)

        filtered_F = filtered_F[:balance_point+1]

        decision_seq.append(most_common)
        preference_seq.append(prefs)

        if len(most_common) == 1:
            break

    return F, decision_seq, preference_seq


def mcdm(F, obj_weights, types):
    cvalues = COMET.make_cvalues(F)
    expert_function = MethodExpert(TOPSIS(), obj_weights, types)
    bounds = SPOTIS.make_bounds(F)

    # method_names = ["TOPSIS", "MABAC", "COMET", "SPOTIS"]
    # print("method_names:", method_names)
    methods = [TOPSIS(), MABAC(), COMET(cvalues, expert_function), SPOTIS(bounds)]

    prefs = []
    ranks = []
    for method in methods:
        pref = method(F, obj_weights, types) # .tolist()
        rank = method.rank(pref) # .tolist()
        prefs.append(pref)
        ranks.append(rank)

    return ranks, prefs


def ranks2decision(ranks):
    most_common = Counter(np.argmin(ranks, axis=1)).most_common()
    most_common = sorted(most_common, key=lambda _: (_[1], _[0]), reverse=True)
    balance_point = most_common[0][0]
    return balance_point, most_common
