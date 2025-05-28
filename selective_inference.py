import numpy as np
from func_timeout import func_timeout, FunctionTimedOut
from parametric_si import parametric_sfs_si
from si4pipeline import (construct_pipelines, initialize_dataset, stepwise_feature_selection, PipelineManager)

def sfs_si(timeout, *args, **kwargs):
    try:
        return func_timeout(timeout, func=parametric_sfs_si, args=args, kwargs=kwargs)
    except FunctionTimedOut:
        return None

def stepwise_selective_inference(support_size) -> PipelineManager:
    return construct_pipelines(stepwise_feature_selection(*initialize_dataset(), support_size))

def subset_fdr(p_values):
    fdr = -np.mean(np.log(1-np.array(p_values)))
    return fdr

def forward_stop_rule(p_values, alpha):
    fdr = np.log(1-np.array(p_values))
    fdr = np.cumsum(fdr)
    for i in range(len(fdr)):
        fdr[i] = -fdr[i]/(i+1)
    stop_at = max(np.where(fdr <= alpha)[0])
    return stop_at, p_values

