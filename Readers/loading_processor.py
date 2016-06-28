__author__ = 'MW'

import os
import theano
import numpy as np
from sklearn.cross_validation import train_test_split
from ecg_provider import DataProvider
from smote.data_resampler import SMOTE
import cPickle
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

DATA_BASES = ['mitdb', 'incartdb']


class DataLoader(object):
    def __init__(self, db_path, data_bases=None, split_factor=0.2,
                  window=256, start=0, stop=600):

        ''' Loads the dataset

        :type db_path: string
        :param db_path: the path to the dataset

        :type file_name: string
        :param file_name: the path to already read data stored in cPickle

        :type write_data: bool
        :param write_data: if true then write data to cPickle file

        :type read_data: bool
        :param read_data: if true then read data from cPickle file

        :type split_factor: float
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
        self.start = start
        self.stop = stop

        self.train_set = [[], [], []]
        self.valid_set = [[], [], []]
        self.test_set = [[], [], []]

    def shared_dataset(self, data_xy, borrow=True):

        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x_qrs, data_x_rr, data_y = data_xy
        data_x_qrs = np.asarray(data_x_qrs.tolist(), dtype=theano.config.floatX)

        data_x_qrs = np.expand_dims(data_x_qrs, axis=1)
        shared_x_qrs = theano.shared(data_x_qrs,
                                    borrow=borrow)

        data_x_rr = np.asarray(data_x_rr.tolist(), dtype=theano.config.floatX)

        data_x_rr = np.expand_dims(data_x_rr, axis=1)
        shared_x_rr = theano.shared(data_x_rr,
                                    borrow=borrow)

        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=np.int32),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x_qrs, shared_x_rr, shared_y

    def load_data(self):
        #############
        # LOAD DATA #
        #############
        print '... loading data from datasets'
        for data_base in self.data_bases:
            if data_base == 'mitdb':
                analysed_channels = [0, 1]
            else:
                analysed_channels = [1, 7]
            data_base_path = os.path.join(self.db_path, data_base)
            records = sorted([record for record in os.listdir(data_base_path) if record.endswith('.dat')])
            dp = DataProvider(data_base_path, split_factor=self.split_factor,
                              window=self.window, start=self.start, stop=self.stop,
                              number_of_channel_to_analyse=None, channels_to_analyse=None)
            for record in records:
                if record.endswith('.dat'):
                    for channel in analysed_channels:
                        record = record.split('.')[0]
                        print 'loading file: %s, channel: %d' % (record, channel)
                        dp.prepare_signal(record, channel)
                        feats, clss = dp.get_data()
                        train_features, \
                        test_valid_features, \
                        train_classes, \
                        test_valid_classes = train_test_split(feats,
                                                              clss,
                                                              train_size=self.split_factor,
                                                              random_state=1234)

                        test_features, \
                        valid_features, \
                        test_classes, \
                        valid_classes = train_test_split(test_valid_features,
                                                         test_valid_classes,
                                                         test_size=0.5,
                                                         random_state=1234)
                        #train_small_set = dp.get_training_set()
                        #print 'train small set: ', len(train_small_set[0])
                        train_features = zip(*train_features)
                        self.train_set[0] += train_features[0]
                        self.train_set[1] += train_features[1]
                        self.train_set[2] += train_classes

                        valid_features = zip(*valid_features)
                        #valid_small_set = dp.get_validate_set()
                        self.valid_set[0] += valid_features[0]
                        self.valid_set[1] += valid_features[1]
                        self.valid_set[2] += valid_classes

                        test_features = zip(*test_features)
                        #test_small_set = dp.get_testing_set()
                        self.test_set[0] += test_features[0]
                        self.test_set[1] += test_features[1]
                        self.test_set[2] += test_classes

        #rr_anomally_class = np.asarray(self.train_set[1])[np.asarray(self.train_set[2])==2]
        #qrs_anomally_class = np.asarray(self.train_set[0])[np.asarray(self.train_set[2])==2]

        self._reshuffle_data()
        test_set_x_qrs, test_set_x_rr, test_set_y = self.shared_dataset(self.test_set)
        valid_set_x_qrs, valid_set_x_rr, valid_set_y = self.shared_dataset(self.valid_set)
        train_set_x_qrs, train_set_x_rr, train_set_y = self.shared_dataset(self.train_set)
        rval = [(train_set_x_qrs, train_set_x_rr, train_set_y), (valid_set_x_qrs, valid_set_x_rr, valid_set_y),
                (test_set_x_qrs, test_set_x_rr, test_set_y)]
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
    dl = DataLoader('/home/marcin/data/', data_bases=['mitdb'], stop=100)
    aa = dl.load_data()
