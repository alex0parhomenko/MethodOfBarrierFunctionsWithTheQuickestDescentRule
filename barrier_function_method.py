import numpy as np
from fast_descent_method import fast_descent_method
from helpers import FuncWithGradient

def get_helper_function(fn : FuncWithGradient, restriction_functions, rk) -> FuncWithGradient:
    return FuncWithGradient(lambda x: fn.get_func(x) - rk * sum(map(lambda g: 1. / g.get_func(x), restriction_functions)),
                            lambda x: fn.get_gradient(x) - rk * sum(map(lambda g: -g.get_gradient(x) / (g.get_func(x) ** 2), restriction_functions)))

def is_method_finish(point, rk, restriction_functions, eps = 0.00000000001) -> bool:
    P = -rk * sum(map(lambda g: 1. / g.get_func(point), restriction_functions))
    return True if np.abs(P) < eps else False

def barrier_function_method(func : FuncWithGradient, restriction_functions, start_point, r0, C):
    iter_num = 0
    rk = r0
    now_point = start_point
    is_continue = True
    while is_continue:
        helper_func = get_helper_function(func, restriction_functions, rk)
        best_point = fast_descent_method(helper_func, now_point, restriction_functions)
        print('Current point: %s', best_point)
        is_continue = not is_method_finish(best_point, rk, restriction_functions)

        if is_continue:
            rk = rk / C
            now_point = best_point
            iter_num += 1

    return now_point