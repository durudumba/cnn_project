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

        for n in range(inputs.shape[0]):
            for c in range(inputs.shape[3]):
                for h in range(s):
                    for w in range(s):
                        block = inputs[n, (w*self.strides) : (w*self.strides)+self.pshape, (h*self.strides) : (h*self.strides)+self.pshape, c]
                        outputs[n, w, h, c] = np.max(block)

                        maxpos = [n, int(np.argmax(block) / self.pshape), (np.argmax(block) % self.pshape), c]

                        self.cached_data.append(maxpos)

        return (outputs, outputs)

    def get_activation_grad(self, z, upstream_gradient):
        # There is no activation function
        return upstream_gradient

    def backward(self, layer_err):

        s = int(layer_err.shape[1] * self.strides)

        dx = np.zeros((layer_err.shape[0], s, s, layer_err.shape[3]))

        for n in range(layer_err.shape[0]):
            for c in range(layer_err.shape[3]):
                for h in range(layer_err.shape[1]):
                    for w,i in zip(range(layer_err.shape[1]), self.cached_data):
                        dx[i] = layer_err[n, w, h, c]

        #최대값 위치를 제외하고 0, 최대값 위치는 역전파로 정해진 값으로 채움
        return dx

    def get_grad(self, inputs, layer_err):
        return 0.

    def update(self, grad, lr):
        pass
        return None

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################