import numpy as np
from helpers import FuncWithGradient
from barrier_function_method import barrier_function_method
from settings import F, restriction_funcitons, r0, C, start_point

def main():
    res = barrier_function_method(F, restriction_funcitons, start_point, r0, C)
    return 0

if __name__ == '__main__':
    main()