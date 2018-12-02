from scipy.optimize import minimize_scalar
import numpy as np
from helpers import FuncWithGradient
from scipy.optimize import Bounds

def mininize(fn : FuncWithGradient, x, restriction_functions) -> float:
    '''
    We supose what we optimize our function on convex closed set
    :param fn: function for optimize, lambda function on one argument
    :param x: start point for optimize
    :param restriction_functions: functions each of must have check_point method which detect is point in posible set
    :return: point with local minimum of function with given restrictions
    '''
    grad = fn.get_gradient(x)
    l, r = 0, 1e9
    while r - l > 0.001:
        mid = (r + l) / 2.
        if (all([f.check_point(x - mid * grad) for f in restriction_functions])):
            l = mid
        else:
            r = mid
    left_border, right_border = 0, l
    if left_border > right_border:
        left_border, right_border = right_border, left_border
    print("Left right borders: {}, {} gradient: {}".format(left_border, right_border, str(grad)))

    val = minimize_scalar(lambda l: fn.get_func(x - l * grad), bounds=[left_border, right_border], method='bounded').x
    return x - val * grad

def fast_descent_method(fn : FuncWithGradient, start_point, restriction_functions, eps=0.00001) -> float:
    return mininize(fn, start_point, restriction_functions)