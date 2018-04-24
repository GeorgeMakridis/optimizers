import numpy as np
from scipy.special import gamma


class CostFunction:
    """
    CostFunction is a class implementing the objective function.
    """

    def __init__(self, features):
        self.features = features

    def __a1(self, features, r, alpha):
        return np.log(gamma(features['x'] + r)) + \
               r * np.log(alpha) - np.log(gamma(r))

    def __a2(self, features, a, b):
        return np.log(gamma(a + b)) + np.log(gamma(b + features['x'])) - np.log(gamma(b)) - np.log(gamma(a + b + features['x']))

    def __a3(self, features, r, alpha):
        return -(r + features['x']) * np.log(alpha + features['T'])

    def __a4(self, features, r, alpha, a, b):
        return np.log(a) - np.log(b + features['x'] - 1) - (r + features['x']) * np.log(alpha + features['tx'])

    def neg_log_likelihood(self, features, params, norm_factor):
        """
        The function computing the negative log likelihood function as given.
        :param features: pandas DataFrame
        :param params: a list of function parameters [r,alpha,a,b]
        :return: float or inf if any of the parameters is negative
        :raise exception if length of parameters list ig greater than 4
        """
        if len(params) > 4:
            raise Exception('The length of params is greater than 4')

        for param in params:
            if param <= 0:
                return +float('inf')

        r = params[0]
        alpha = params[1]
        a = params[2]
        b = params[3]

        alpha = alpha * norm_factor

        N = features.shape[0]

        delta = 1 & features['x']

        results = np.sum(
            (-1)*(self.__a1(features, r, alpha) + self.__a2(features, a, b) + np.log(np.exp(self.__a3(features, r, alpha)) +
                                                                                 delta * np.exp(
                                                                                     self.__a4(features, r, alpha, a,
                                                                                               b))))) / N

        return results