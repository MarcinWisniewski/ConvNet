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

N_KERNS = (6, 6, 12, 12, 10)
# dict from class to wfdb code
annotation_dict = {0: 0, 1: 1, 2: 5, 3: 9}
db_path = '/home/marcin/data/mitdb'

files = os.listdir(db_path)
files = [record.split('.')[0] for record in files if record.split('.')[-1] == 'dat']


def recognize_signal():
    x = T.matrix('x', dtype=theano.config.floatX)
    batch_size = 1
    rng = np.random.RandomState(23455)
    f = open('model.bin', 'rb')
    cn_net = CNN(x, N_KERNS, batch_size)
    cn_net.__setstate__(cPickle.load(f))
    f.close()
    test_prediction = lasagne.layers.get_output(cn_net.network, deterministic=True)
    window = 1024
    for record in files:
        print '...analysing record', record
        total_time = timeit.default_timer()
        file_path = os.path.join(db_path, record)
        dp = DataProvider(file_path, split_factor=100, window=window, step=window)
        dp.prepare_signal()
        signal = dp.signal
        inputMatrix = dp.inputMatrix
        #reference_annots = dp.annots
        print 'signal length', len(signal)
        get_r_peaks = theano.function([x], test_prediction)
        start_position = 0
        end_position = len(signal)
        for index_of_frame in xrange(len(inputMatrix)):
            input_matrix = np.expand_dims(inputMatrix[index_of_frame], 0)
            indexes = get_r_peaks(input_matrix)
            indexes *= window
            indexes += index_of_frame*window  #window/skip






        print 'saving annot file'
        wrann(annot_list, file_path +'.tan')
        print timeit.default_timer() - total_time


if __name__ == '__main__':
    #cProfile.run('recognize_signal()')
    recognize_signal()
