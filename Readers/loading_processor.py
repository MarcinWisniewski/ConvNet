__author__ = 'MW'

import os
import theano
import numpy
from ecg_provider import DataProvider
import matplotlib.pyplot as plt
import cPickle


def load_data(db_path, file_name='data.bin', write_data=False, read_data=False):

    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

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

    #############
    # LOAD DATA #
    #############
    if read_data:
        print "...reading data from: ", file_name
        file_h = open(file_name, 'rb')
        test_set, valid_set, train_set = cPickle.load(file_h)
        file_h.close()
    else:
        print '... loading data from datasets'
        files = ['100', '101', '103', '105', '106', '107',
                # '108', '109', '111', '112', '113', '114',
                #'115', '116', '117', '118', '119', '121',
                # '122', '123', '124', '200', '201', '202',
                # '203', '205', '207', '208', '209', '210',
                 #'212', '213', '214', '215', '217', '219',
                 #'220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
        #files = ['100', '119']
        #files = ['100']

        train_set = [[], []]
        valid_set = [[], []]
        test_set = [[], []]
        for file in files:
            print 'loading file: ', file
            dp = DataProvider(os.path.join(db_path, file), split_factor=85,
                              window=1024, start=0, stop=500)
            dp.prepare_signal()
            dp.reshuffle_data()
            train_small_set = dp.getTrainingSet()
            train_set[0] += train_small_set[0]
            train_set[1] += train_small_set[1]

            valid_small_set = dp.getValidateSet()
            valid_set[0] += valid_small_set[0]
            valid_set[1] += valid_small_set[1]

            test_small_set = dp.getTestingSet()
            test_set[0] += test_small_set[0]
            test_set[1] += test_small_set[1]

    if write_data:
        print '...saving data to: ', file_name
        file_h = open(file_name, 'wb')
        cPickle.dump((test_set, valid_set, train_set), file_h, protocol=cPickle.HIGHEST_PROTOCOL)
        file_h.close()
        print 'Done!!'
    else:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)
        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        return rval

if __name__ == '__main__':
    load_data(file_name='data_3.bin', write_data=True)
