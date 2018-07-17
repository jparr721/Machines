import numpy as np


class Perceptron(object):
    """
    Perceptron classifier

    Params
    ------
    eta: float
        Learning rate (usually between 0.0 and 1.0)
    n_iter: int
        Number of passes over the data set
    random_state: int
        Random number for detemrination of initial weight values

    Attributes
    ---------
    w_: list
        The weights after fitting
    errors_: list
        Number of misclassifications in each epoch (iteration)
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the training data
        """
        rgen = np.random.RandomState(self.random_state)

        #  Fill vector with random values from the normal distribution
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                #  Bias unit
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        Calculate net input
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        Predict class label after unit step
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
