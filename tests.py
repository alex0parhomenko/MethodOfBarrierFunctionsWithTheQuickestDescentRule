from fast_descent_method import fast_descent_method
from helpers import FuncWithGradient
import numpy as np
from barrier_function_method import barrier_function_method

def test_fast_descent_method():
    fn = FuncWithGradient(lambda x: x ** 2, lambda x: 2 * x)
    eps=0.00001
    result_x = 0
    assert abs(fast_descent_method(fn, 3, []) - result_x) < eps
    assert abs(fast_descent_method(fn, 4, []) - result_x) < eps

def test_barrier_function_method():
    fn = FuncWithGradient(lambda x: 8 * np.sin(x) + x**2 - x**3, lambda x: (2 - 3 * x) * x + 8 * np.cos(x))
    g1 = FuncWithGradient(lambda x: -x**2 + 4, lambda x: -2 * x)
    start_point = 1.0
    eps = 0.01
    answer = -0.953
    res = barrier_function_method(fn, [g1], start_point, 0.5, 2)
    assert abs(res - answer < eps)