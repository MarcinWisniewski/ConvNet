__author__ = 'MW'

import os
import theano
import numpy
from ecg_provider import DataProvider
import matplotlib.pyplot as plt
import cPickle

DATA_BASES = ['mitdb', 'incartdb']


def load_data(db_path, data_bases=None, split_factor=85,
              window=1024, step=1024, start=0, stop=600):

    ''' Loads the dataset

    :type db_path: string
    :param db_path: the path to the dataset

    :type file_name: string
    :param file_name: the path to already read data stored in cPickle

    :type write_data: bool
    :param write_data: if true then write data to cPickle file

    :type read_data: bool
    :param read_data: if true then read data from cPickle file

    :type split_factor: int
    :param split_factor: percentage of split: train data / test and validadion data

    :type window: int
    :param window: size of feature vector

    :type start: int
    :param: start: start time in sec to wfdb reader

    :type stop: int
    :param: stop: stop time in sec to wfdb reader

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
        data_x = numpy.asarray(data_x, dtype=theano.config.floatX)
        data_x = numpy.expand_dims(data_x, axis=1)
        shared_x = theano.shared(data_x,
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
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

    print '... loading data from datasets'

    train_set = [[], []]
    valid_set = [[], []]
    test_set = [[], []]

    if data_bases is None:
        data_bases = DATA_BASES
    for data_base in data_bases:
        data_base_path = os.path.join(db_path, data_base)
        records = os.listdir(data_base_path)
        for record in records:
            if record.endswith('.dat'):
                print 'loading file: ', record
                record = record.split('.')[0]
                dp = DataProvider(os.path.join(data_base_path, record), split_factor=split_factor,
                                  window=window, step=step, start=start, stop=stop)
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

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

if __name__ == '__main__':
    load_data('/home/marcin/data/', data_bases=['mitdb', 'incartdb'], stop=1000)
