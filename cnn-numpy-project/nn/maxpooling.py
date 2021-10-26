import numpy as np
from nn.layer import AbstractLayer

class MaxPooling(AbstractLayer):
    def __init__(self, pshape, strides=2):
        self.pshape = pshape  # pooling shape (height * width)
        self.strides = strides
        self.cached_data = []

    ###########################################################################
    # TODO: Implement the Max-pooling layer                                   #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, inputs):

        s = int(inputs.shape[1] / self.strides)

        outputs = np.zeros((inputs.shape[0], s, s, inputs.shape[3]))

        for h in range(s):
            for w in range(s):
                outputs[:, w, h, :] = np.max(inputs[:, (w*self.strides) : (w*self.strides)+self.pshape, (h*self.strides) : (h*self.strides)+self.pshape, :])

        return (outputs, outputs)

    def get_activation_grad(self, z, upstream_gradient):
        # There is no activation function
        return upstream_gradient

    def backward(self, layer_err):




        #최대값 위치를 제외하고 0, 최대값 위치는 역전파로 정해진 값으로 채움
        return None

    def get_grad(self, inputs, layer_err):
        pass
        return None

    def update(self, grad, lr):
        pass
        return None

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################