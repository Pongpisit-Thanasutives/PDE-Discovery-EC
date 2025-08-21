import numpy as np
from collections import Counter
from pymcdm import weights as obj_w
from pymcdm.methods import TOPSIS, MABAC, COMET, SPOTIS
from pymcdm.methods.comet_tools import MethodExpert


def compromise_programming(best_subsets, dataset):
    XX, yy = dataset
    F = np.array(
        [[np.linalg.lstsq(XX[:, efi], yy)[1][0], len(efi)] for efi in best_subsets]
    )
    nF = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))

    obj_weights = obj_w.entropy_weights(F)
    types = [-1, -1]
    cvalues = COMET.make_cvalues(F)
    expert_function = MethodExpert(TOPSIS(), obj_weights, types)
    bounds = SPOTIS.make_bounds(F)

    method_names = ["TOPSIS", "MABAC", "COMET", "SPOTIS"]
    methods = [TOPSIS(), MABAC(), COMET(cvalues, expert_function), SPOTIS(bounds)]

    ranks = [method.rank(method(F, obj_weights, types)) for method in methods]
    ranks = sorted(Counter(np.argmin(ranks, axis=1)).most_common())
    balance_point = F[ranks[0][0]]

    return balance_point, ranks
