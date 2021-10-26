import numpy as np

class SGDOptimizer():
    def __init__(self):
        pass

    def update(self, dx, lr = 0.001):
        update_value = dx * lr
        return update_value


class MomentumSGDOptimizer():
    def __init__(self):
        # parameters for Adam
        self.beta = 0.9
        self.v = 0

    def update(self, dx, lr = 0.001):
        """
        A implementation of the Momentum SGD optimizer.

        Input:
        - dx: gradient of the target weight
        - lr: learning rate

        Returns a tuple of:
        - out: update_value
        """
        ###########################################################################
        # TODO: Implement the Momentum SGD optimizer                              #
        ###########################################################################

        #update_value = None

        self.v = self.beta * self.v - lr * dx
        dx += self.v

        update_value = dx

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return update_value


class AdamOptimizer():
    def __init__(self):
        # parameters for Adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.m = 0.0
        self.v = 0.0
        self.eps = 1e-8
        self.t = 0

    def update(self, dx, lr = 0.001):
        """
        A implementation of the Adam optimizer.

        Input:
        - dx: gradient of the target weight
        - lr: learning rate

        Returns a tuple of:
        - out: update_value
        """
        ###########################################################################
        # TODO: Implement the Adam optimizer                                      #
        ###########################################################################

        #update_value = None
        self.t += 1
        self.m = self.beta1 * self.m + lr * dx
        self.v = self.beta2 * self.v + (1 - self.beta2) * dx * dx
        first_unbias = self.m / (1 - self.beta1 ** self.t)
        second_unbias = self.v / (1 - self.beta2 ** self.t)
        update_value = lr * first_unbias / (np.sqrt(second_unbias) + self.eps)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return update_value