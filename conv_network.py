__author__ = 'user'

from conv_layer import LeNetConvPoolLayer
from mlp import HiddenLayer
from logistic_sgd import LogisticRegression
import theano.tensor as T


class CNN(object):
    def __getstate__(self):
        weights = [parameter.get_value() for parameter in self.params]
        return weights

    def __setstate__(self, weights):
        weight = iter(weights)
        for parameter in self.params:
            parameter.set_value(weight.next())

    def __init__(self, rng, input, n_kerns, batch_size):
        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        self.input_signal_length = 1024
        self.layer0_input = input.reshape((batch_size, 1, 1, self.input_signal_length))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)

        self.layer0_pooling_factor = 2
        self.layer0_filter_length = 71
        print 'layer 0 input: ', self.input_signal_length
        self.layer0 = LeNetConvPoolLayer(
            rng,
            input=self.layer0_input,
            image_shape=(batch_size, 1, 1, self.input_signal_length),
            filter_shape=(n_kerns[0], 1, 1, self.layer0_filter_length),
            poolsize=(1, self.layer0_pooling_factor)
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        self.layer1_pooling_factor = 2
        self.layer1_filter_length = 36
        self.layer1_input_length = (self.input_signal_length-self.layer0_filter_length+1)/self.layer0_pooling_factor
        print 'layer 1 input: ', self.layer1_input_length
        self.layer1 = LeNetConvPoolLayer(
            rng,
            input=self.layer0.output,
            image_shape=(batch_size, n_kerns[0], 1, self.layer1_input_length),
            filter_shape=(n_kerns[1], n_kerns[0], 1, self.layer1_filter_length),
            poolsize=(1, self.layer1_pooling_factor)
        )

        self.layer2_pooling_factor = 2
        self.layer2_filter_length = 35
        self.layer2_input_length = (self.layer1_input_length - self.layer1_filter_length + 1) / self.layer1_pooling_factor
        print 'layer 2 input: ', self.layer2_input_length
        self.layer2 = LeNetConvPoolLayer(
            rng,
            input=self.layer1.output,
            image_shape=(batch_size, n_kerns[1], 1, self.layer2_input_length),
            filter_shape=(n_kerns[2], n_kerns[1], 1, self.layer2_filter_length),
            poolsize=(1, self.layer2_pooling_factor)
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        self.layer3_input = self.layer2.output.flatten(2)
        #self.layer3_input_length = layer2_input_length / layer2_pooling_factor
        self.layer3_input_length = 93
        print 'layer 2 output', self.layer3_input_length
        print 'layer 3 input: ', n_kerns[2] * 1 * self.layer3_input_length
        print 'layer 3 output: ', self.layer3_input_length / 2
        # construct a fully-connected sigmoidal layer
        self.layer3 = HiddenLayer(
            rng,
            input=self.layer3_input,
            n_in=n_kerns[2] * 1 * self.layer3_input_length,
            n_out=self.layer3_input_length / 2,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        self.layer4_input_length = self.layer3_input_length / 2
        print 'layer 4 input: ', self.layer4_input_length
        self.layer4 = LogisticRegression(input=self.layer3.output,
                                    n_in=self.layer4_input_length,
                                    n_out=6)

        self.errors = self.layer4.errors

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.layer4.params + self.layer3.params + \
                      self.layer2.params + self.layer1.params + self.layer0.params
