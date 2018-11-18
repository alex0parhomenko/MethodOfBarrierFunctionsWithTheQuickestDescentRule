import  numpy as np
from helpers import FuncWithGradient

F1=FuncWithGradient(lambda x: np.sin(x) + x ** 3, lambda x: np.cos(x) + 3 * x ** 2)
G1 = FuncWithGradient(lambda x: x ** 2 - 3, lambda x: 2 * x)
G2 = FuncWithGradient(lambda x: x - 4, lambda x: 1)

r0 = 0.3
C = 2