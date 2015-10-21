"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression
from loading_processor import load_data
from mlp import HiddenLayer
from conv_layer import LeNetConvPoolLayer
import matplotlib.pyplot as plt

def evaluate_lenet5(learning_rate=0.1, n_epochs=10,
                    nkerns=[10, 20, 30], batch_size=5000):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    print 'training example shape: ', train_set_x.get_value(borrow=True).shape
    print 'testing example shape: ', test_set_x.get_value(borrow=True).shape
    print 'valid example shape: ', valid_set_x.get_value(borrow=True).shape
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    input_signal_length = 1024
    layer0_input = x.reshape((batch_size, 1, 1, input_signal_length))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)

    layer0_pooling_factor = 2
    layer0_filter_length = 71
    print 'layer 0 input: ', input_signal_length
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 1, input_signal_length),
        filter_shape=(nkerns[0], 1, 1, layer0_filter_length),
        poolsize=(1, layer0_pooling_factor)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1_pooling_factor = 2
    layer1_filter_length = 36
    layer1_input_length = (input_signal_length-layer0_filter_length+1)/layer0_pooling_factor
    print 'layer 1 input: ', layer1_input_length
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 1, layer1_input_length),
        filter_shape=(nkerns[1], nkerns[0], 1, layer1_filter_length),
        poolsize=(1, layer1_pooling_factor)
    )

    layer2_pooling_factor = 2
    layer2_filter_length = 35
    layer2_input_length = (layer1_input_length - layer1_filter_length + 1) / layer1_pooling_factor
    print 'layer 2 input: ', layer2_input_length
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 1, layer2_input_length),
        filter_shape=(nkerns[2], nkerns[1], 1, layer2_filter_length),
        poolsize=(1, layer2_pooling_factor)
    )


    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer3_input = layer2.output.flatten(2)
    #layer3_input_length = layer2_input_length / layer2_pooling_factor
    layer3_input_length = 93
    print 'layer 2 output', layer3_input_length
    print 'layer 3 input: ', nkerns[2] * 1 * layer3_input_length
    print 'layer 3 output: ', layer3_input_length / 2
    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * 1 * layer3_input_length,
        n_out=layer3_input_length / 2,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3_input_length = layer3_input_length / 2
    print 'layer 4 input: ', layer3_input_length
    layer4 = LogisticRegression(input=layer3.output,
                                n_in=layer3_input_length,
                                n_out=6)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_labels = T.cast(theano.shared(numpy.asarray(numpy.zeros(train_set_x.get_value(borrow=True).shape[0]), dtype=theano.config.floatX)), 'int32')
    input = T.tensor4('input')
    predict = theano.function(
        [index],
        layer4.output(),
        givens={x: train_set_x[index:],
                y: test_labels},
        on_unused_input='warn'
    )
    # end-snippet-1


    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 1000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
    end_time = timeit.default_timer()
    plt.figure(1)
    for i in range(nkerns[0]):
        W = layer0.W.container.storage[0][i][0][0]
        H = numpy.fft.fft(W, 1024)
        plt.subplot(nkerns[0], 1, i+1)
        plt.plot(20*numpy.log10(numpy.abs(H[0:512])))

    plt.figure(2)
    for i in range(nkerns[1]):
        W = layer1.W.container.storage[0][i][0][0]
        H = numpy.fft.fft(W, 1024)
        plt.subplot(nkerns[1], 1, i+1)
        plt.plot(20*numpy.log10(numpy.abs(H[0:512])))
    plt.show()

    RET = []
    train_set_x = train_set_x.get_value
    for it in range(len(train_set_x)):
        test_data = train_set_x[it]
        N = len(test_data)
        test_data = theano.shared(numpy.asarray(test_data, dtype=theano.config.floatX))
        # just zeroes
        test_labels = T.cast(theano.shared(numpy.asarray(numpy.zeros(batch_size), dtype=theano.config.floatX)), 'int32')

        ppm = theano.function([index], layer4.pred_probs(),
            givens={
                x: test_data[index * batch_size: (index + 1) * batch_size],
                y: test_labels
            }, on_unused_input='warn')

        # p : predictions, we need to take argmax, p is 3-dim: (# loop iterations x batch_size x 2)
        p = [ppm(ii) for ii in range( N // batch_size)]
        #p_one = sum(p, [])
        #print p
        p = numpy.array(p).reshape((N, 10))
        print (p)
        p = numpy.argmax(p, axis=1)
        p = p.astype(int)
        RET.append(p)
    print RET

    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
