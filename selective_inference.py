from parametric_si import parametric_sfs_si
from func_timeout import func_timeout, FunctionTimedOut

def sfs_si(timeout, *args, **kwargs):
    try:
        return func_timeout(timeout, func=parametric_sfs_si, args=args, kwargs=kwargs)
    except FunctionTimedOut:
        return None

