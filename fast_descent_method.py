from scipy.optimize import minimize_scalar
import numpy as np
from helpers import FuncWithGradient

def mininize(fn : FuncWithGradient, x) -> float:
    l_min = minimize_scalar(lambda l: fn.get_func(x - l * fn.get_gradient(x))).x
    return x - l_min * fn.get_gradient(x)

def fast_descent_method(fn : FuncWithGradient, start_point, eps=0.0001) -> float:
    prev_point = start_point
    current_point = start_point
    iter = 0
    while iter == 0 or np.linalg.norm(prev_point - current_point) > eps:
        prev_point = current_point
        current_point = mininize(fn, start_point)
        iter += 1
    return current_point