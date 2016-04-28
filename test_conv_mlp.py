__author__ = 'MW'

import pickle as cPickle
import timeit

import numpy as np
import theano.tensor as T
import theano
import lasagne
import os
import matplotlib.pyplot as plt

from CNN.conv_network import CNN
from Readers.ecg_provider import DataProvider
from WFDBTools.wfdb_wrann import wrann
import cProfile

N_KERNS = (32, 32, 32, 32, 32)
# dict from class to wfdb code
annotation_dict = {0: 0, 1: 1, 2: 5, 3: 9}
db_path = '/home/marcin/data/mitdb'

files = os.listdir(db_path)
files = [record.split('.')[0] for record in files if record.split('.')[-1] == 'dat']
SHOW_FRAME = False


def recognize_signal():
    x = T.tensor4('x', dtype=theano.config.floatX)    # the data is presented as rasterized images
    batch_size = 128
    rng = np.random.RandomState(23455)
    f = open('qrs_model.bin', 'rb')
    cn_net = CNN(x, N_KERNS, batch_size)
    cn_net.__setstate__(cPickle.load(f))
    f.close()
    test_prediction = lasagne.layers.get_output(cn_net.network, deterministic=True)
    window = 512
    dp = DataProvider(db_path, split_factor=100,
                      window=window, step=window,
                      number_of_channel_to_analyse=1,
                      channels_to_analyse=[0])
    get_r_peaks = theano.function([x], test_prediction)
    for record in files:
        print '...analysing record', record
        total_time = timeit.default_timer()
        dp.prepare_signal(record)
        signal = dp.signal
        feature_matrix = dp.feature_matrix
        print 'signal length', len(signal)
        annot_list = []
        for index_of_frame in xrange(0, len(feature_matrix)-batch_size, batch_size):
            input_matrix = np.asarray(feature_matrix[index_of_frame:index_of_frame+batch_size],
                                      dtype=theano.config.floatX)
            input_matrix = np.expand_dims(input_matrix, 1)
            indexes = get_r_peaks(input_matrix)
            indexes *= window
            for mini_batch_index in xrange(batch_size):
                if SHOW_FRAME:
                    plt.plot(input_matrix[mini_batch_index][0][0])
                    plt.plot(indexes[mini_batch_index], 0.9, 'ro')
                    plt.close()

                indexes[mini_batch_index] += (index_of_frame+mini_batch_index)*window  #window/skip

            annot_list += [(int(index), 1) for index in indexes if index > 0]

        print 'saving annot file'
        file_path = os.path.join(db_path, record)
        wrann(annot_list, file_path + '.tan')
        print timeit.default_timer() - total_time


if __name__ == '__main__':
    #cProfile.run('recognize_signal()')
    recognize_signal()
