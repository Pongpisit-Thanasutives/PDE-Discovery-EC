from parametric_si import parametric_sfs_si
from func_timeout import func_timeout, FunctionTimedOut

def func_with_timeout(func, args, timeout=1):
    try:
        return func_timeout(timeout, func, args=args)
    except FunctionTimedOut:
        return None

def sfs_si(timeout, *args):
    return func_with_timeout(func=parametric_sfs_si, args=args, timeout=timeout)

