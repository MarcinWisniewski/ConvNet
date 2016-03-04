import os
import sys
import timeit
import numpy
import lasagne
import theano
import theano.tensor as T
import pickle as cPickle
from Readers.loading_processor import load_data
from CNN.conv_network import CNN


def evaluate_ecg_net(learning_rate=0.001, momentum=0.9, n_epochs=20,
                    n_kerns=(24, 16, 16, 16, 16), batch_size=256, use_model=True):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (wfdb here)

    :type n_kerns: list of ints
    :param n_kerns: number of kernels on each layer

    :type batch_size: int
    :param batch_size: number of examples in minibatch
    """

    rng = numpy.random.RandomState(23455)
    db_path = '/home/marcin/data/mitdb/'
    records = ['100', '101', '103', '100', '119', '232', '217']

    data_sets = load_data(db_path, stop=1400)
    train_set_x, train_set_y = data_sets[0]
    valid_set_x, valid_set_y = data_sets[1]
    test_set_x, test_set_y = data_sets[2]

    # compute number of minibatches for training, validation and testing
    print 'number of training examples: ', train_set_x.get_value(borrow=True).shape[0]
    print 'number of testing examples: ', test_set_x.get_value(borrow=True).shape[0]
    print 'number of valid examples: ', valid_set_x.get_value(borrow=True).shape[0]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x', dtype=theano.config.floatX)   # the data is presented as rasterized images
    y = T.matrix('y', dtype=theano.config.floatX)   # the target is a 2D matrix with at most 10 normalised indexes of qrs

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    cnn = CNN(x, n_kerns, batch_size)
    print 'CNN with %i parameters' % lasagne.layers.count_params(cnn.network)
    prediction = lasagne.layers.get_output(cnn.network)
    # the cost we minimize during training is the NLL of the model
    loss = lasagne.objectives.squared_error(prediction, y)
    loss = loss.mean()

    parameters = lasagne.layers.get_all_params(cnn.network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, parameters,
                                                learning_rate=learning_rate,
                                                momentum=momentum)

    test_prediction = lasagne.layers.get_output(cnn.network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, y)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        [test_loss],
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        [test_loss],
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_model = theano.function(
        [index],
        loss,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    ###############
    # TRAIN MODEL #
    ###############

    if use_model:
        f = open('qrs_model.bin', 'rb')
        cnn.__setstate__(cPickle.load(f))
        f.close()

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
    best_cnn = CNN(x, n_kerns, batch_size)
    print 'valid frequency - %i iterations' % validation_frequency
    print 'number of epochs - %i ' % n_epochs
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if iter % 10 == 0:
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

                    best_weights = cnn.__getstate__()
                    best_cnn.__setstate__(best_weights)
                    f = open('qrs_model.bin', 'wb')
                    cPickle.dump(best_cnn.__getstate__(), f, protocol=cPickle.HIGHEST_PROTOCOL)
                    f.close()

    end_time = timeit.default_timer()

    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_ecg_net(use_model=True)

