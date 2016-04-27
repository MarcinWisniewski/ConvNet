__author__ = 'MW'

import os
import theano
import numpy as np
from ecg_provider import DataProvider
import matplotlib.pyplot as plt
import cPickle

DATA_BASES = ['mitdb', 'incartdb']


class DataLoader(object):
    def __init__(self, db_path, data_bases=None, split_factor=85,
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

        self.rnd_gen = np.random.seed(2345667)
        self.db_path = db_path
        if data_bases is None:
            self.data_bases = DATA_BASES
        else:
            self.data_bases = data_bases
        self.window = window
        self.split_factor = split_factor
        self.step = step
        self.start = start
        self.stop = stop

        self.train_set = [[], []]
        self.valid_set = [[], []]
        self.test_set = [[], []]

    def shared_dataset(self, data_xy, borrow=True):

        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        data_x = np.asarray(data_x, dtype=theano.config.floatX)

        data_x = np.expand_dims(data_x, axis=1)
        shared_x = theano.shared(data_x,
                                 borrow=borrow)
        data_y = np.expand_dims(data_y, axis=1)
        shared_y = theano.shared(np.asarray(data_y,
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

    def load_data(self):
        #############
        # LOAD DATA #
        #############
        print '... loading data from datasets'
        for data_base in self.data_bases:
            data_base_path = os.path.join(self.db_path, data_base)
            records = [record for record in os.listdir(data_base_path) if record.endswith('.dat')]
            dp = DataProvider(data_base_path, split_factor=self.split_factor,
                              window=self.window, step=self.step, start=self.start, stop=self.stop,
                              number_of_channel_to_analyse=1, channels_to_analyse=[1])
            for record in records:
                if record.endswith('.dat'):
                    print 'loading file: ', record
                    record = record.split('.')[0]
                    dp.prepare_signal(record)
                    train_small_set = dp.get_training_set()
                    print 'train small set: ', len(train_small_set[0])
                    self.train_set[0] += train_small_set[0]
                    self.train_set[1] += train_small_set[1]

                    valid_small_set = dp.get_validate_set()
                    self.valid_set[0] += valid_small_set[0]
                    self.valid_set[1] += valid_small_set[1]

                    test_small_set = dp.get_testing_set()
                    self.test_set[0] += test_small_set[0]
                    self.test_set[1] += test_small_set[1]

        #self._reshuffle_data()
        test_set_x, test_set_y = self.shared_dataset(self.test_set)
        valid_set_x, valid_set_y = self.shared_dataset(self.valid_set)
        train_set_x, train_set_y = self.shared_dataset(self.train_set)
        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        return rval

    def _reshuffle_data(self):
        temp = np.transpose(self.train_set)
        np.random.shuffle(temp)
        self.train_set = np.transpose(temp)

        temp = np.transpose(self.test_set)
        np.random.shuffle(temp)
        self.test_set = np.transpose(temp)

        temp = np.transpose(self.valid_set)
        np.random.shuffle(temp)
        self.valid_set = np.transpose(temp)


if __name__ == '__main__':
    dl = DataLoader('/home/marcin/data/', data_bases=['mitdb'], stop=10)
    aa = dl.load_data()
