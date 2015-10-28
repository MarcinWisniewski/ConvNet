__author__ = 'MW'

import theano
import numpy
from ecg_provider import DataProvider
import matplotlib.pyplot as plt


def load_data():

    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data'
    files = ['100', '101', '103', '106', '119']
    #files = ['100', '119']
    #files = ['15s']

    train_set = [[], []]
    valid_set = [[], []]
    test_set = [[], []]
    for file in files:
        print 'loading file: ', file
        dp = DataProvider('C:\\Users\\user\\data\\mitdb\\'+file, 70, 1024)
        dp.prepare_signal()
        dp.reshuffleData()
        train_small_set = dp.getTrainingSet()
        train_set[0] += train_small_set[0]
        train_set[1] += train_small_set[1]

        valid_small_set = dp.getValidateSet()
        valid_set[0] += valid_small_set[0]
        valid_set[1] += valid_small_set[1]

        test_small_set = dp.getTestingSet()
        test_set[0] += test_small_set[0]
        test_set[1] += test_small_set[1]

    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):

        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype='int32'),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    #plt.hist(train_set[1][1:5000], 6)
    #plt.show()
    return rval

if __name__ == '__main__':
    load_data()
    print 1