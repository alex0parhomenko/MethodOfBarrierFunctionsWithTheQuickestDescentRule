import  numpy as np
from helpers import FuncWithGradient

F=FuncWithGradient(lambda x: 8 * np.sin(x) + x**2 - x**3, lambda x: (2 - 3 * x) * x + 8 * np.cos(x))
G1 = FuncWithGradient(lambda x: x**2 - 4, lambda x: 2 * x)

restriction_funcitons = [G1]
start_point = 1.4
r0 = 3
C = 2