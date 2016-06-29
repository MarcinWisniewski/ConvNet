import os
import sys
import timeit
import numpy
import lasagne
import theano
import theano.tensor as T
import pickle as cPickle
from Readers.loading_processor import DataLoader
from CNN.conv_network import CNN
from theano.compile.nanguardmode import NanGuardMode
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


def evaluate_ecg_net(learning_rate=0.01, momentum=0.9, n_epochs=60,
                     qrs_n_kerns=(50, 65, 30, 32, 8),
                     rr_n_kerns=(45, 64, 50, 32, 8),
                     batch_size=1024, use_model=False):
    """ qrs detector on mit and incart data (fs=360Hz)

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type momentum: float
    :param momentum: momentum in SGD

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type qrs_n_kerns: list of ints
    :param qrs_n_kerns: number of kernels on each layer for qrs

    :type rr_n_kerns: list of ints
    :param rr_n_kerns: number of kernels on each layer for rr


    :type batch_size: int
    :param batch_size: number of examples in minibatch

    :type use_model: bool
    param: use_model: choosing to use pre trained model
    """

    rng = numpy.random.RandomState(23455)
    db_path = '/home/marcin/data/'
    dl = DataLoader(db_path, split_factor=0.80,
                    window=256, start=0, stop=100)

    data_sets = dl.load_data()
    train_set_x_qrs, train_set_x_rr, dummy, train_set_y = data_sets[0]
    valid_set_x_qrs, valid_set_x_rr, dummy, valid_set_y = data_sets[1]
    test_set_x_qrs, test_set_x_rr, dummy, test_set_y = data_sets[2]

    print 'number of training examples: ', train_set_x_qrs.get_value(borrow=True).shape[0]
    print 'number of testing examples: ', test_set_x_qrs.get_value(borrow=True).shape[0]
    print 'number of valid examples: ', valid_set_x_qrs.get_value(borrow=True).shape[0]
    n_train_batches = train_set_x_qrs.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x_qrs.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x_qrs.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x_qrs = T.tensor4('x_qrs', dtype=theano.config.floatX)    # the data is presented as qrs normalized to [0-1]
    x_rr = T.tensor4('x_rr', dtype=theano.config.floatX)    # the data is presented as qrs normalized to [0-1]

    y = T.ivector('y')   # the target is a 2D matrix with 1 normalised index of qrs

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    cnn = CNN(x_qrs, x_rr, qrs_n_kerns, rr_n_kerns, batch_size)
    print 'CNN with %i parameters' % lasagne.layers.count_params(cnn.mlp_net)

    if use_model:
        print 'loading model from file...'
        f = open('qrs_model.bin', 'rb')
        cnn.__setstate__(cPickle.load(f))
        f.close()

    prediction = lasagne.layers.get_output(cnn.mlp_net)
    loss = lasagne.objectives.categorical_crossentropy(prediction, y)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(cnn.mlp_net, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params,
                                                learning_rate=learning_rate,
                                                momentum=momentum)

    test_prediction = lasagne.layers.get_output(cnn.mlp_net, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, y)
    test_loss = test_loss.mean()

    validate_prediction = lasagne.layers.get_output(cnn.mlp_net, deterministic=True)
    validate_loss = lasagne.objectives.categorical_crossentropy(validate_prediction, y)
    validate_loss = validate_loss.mean()

    test_model = theano.function(
        [index],
        [test_loss],
        givens={
            x_qrs: test_set_x_qrs[index * batch_size: (index + 1) * batch_size],
            x_rr: test_set_x_rr[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        [validate_loss],
        givens={
            x_qrs: valid_set_x_qrs[index * batch_size: (index + 1) * batch_size],
            x_rr: valid_set_x_rr[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_model = theano.function(
        [index],
        loss,
        updates=updates,
        givens={
            x_qrs: train_set_x_qrs[index * batch_size: (index + 1) * batch_size],
            x_rr: train_set_x_rr[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

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
    best_cnn = CNN(x_qrs, x_rr, qrs_n_kerns, rr_n_kerns, batch_size)
    print 'valid frequency - %i iterations' % validation_frequency
    print 'number of epochs - %i ' % n_epochs
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if iter % 50 == 0:
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
    evaluate_ecg_net()

