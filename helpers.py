class FuncWithGradient(object):
    def __init__(self, func, gradient):
        self.__func = func
        self.__gradient = gradient

    @property
    def get_func(self):
        return self.__func

    @property
    def get_gradient(self):
        return self.__gradient

    def check_point(self, x):
        return self.__func(x) <= 0