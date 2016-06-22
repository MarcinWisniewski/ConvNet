__author__ = 'MW'

import pickle as cPickle
import timeit

import numpy as np
import theano.tensor as T
import theano
import lasagne
import os
from subprocess import call
import matplotlib.pyplot as plt

from CNN.conv_network import CNN
from Readers.ecg_provider import DataProvider
from WFDBTools.wfdb_wrann import wrann
import cProfile

qrs_n_kerns=(50, 65, 30, 32, 16)
rr_n_kerns=(45, 64, 50, 16)

# dict from class to wfdb code
annotation_dict = {0: 0, 1: 1, 2: 5, 3: 9}
db_path = '/home/marcin/data/mitdb'

files = os.listdir(db_path)
files = [record.split('.')[0] for record in files if record.split('.')[-1] == 'dat']
SHOW_FRAME = False


def recognize_signal():
    base_dir = os.getcwd()
    x_qrs = T.tensor4('x_qrs', dtype=theano.config.floatX)    # the data is presented as qrs normalized to [0-1]
    x_rr = T.tensor4('x_rr', dtype=theano.config.floatX)    # the data is presented as qrs normalized to [0-1]
    batch_size = 128
    rng = np.random.RandomState(23455)
    f = open('qrs_model.bin', 'rb')
    cn_net = CNN(x_qrs, x_rr, qrs_n_kerns, rr_n_kerns, batch_size)
    cn_net.__setstate__(cPickle.load(f))
    f.close()
    test_prediction = lasagne.layers.get_output(cn_net.mlp_net, deterministic=True)
    dp = DataProvider(db_path, split_factor=100, window=256,
                      start=0, stop=-1)
    get_r_peaks = theano.function([x_qrs, x_rr], test_prediction)
    for record in files:
        print '...analysing record', record
        total_time = timeit.default_timer()
        dp.prepare_signal(record)
        signal = dp.signal
        qrs_feature_matrix = dp.qrs_feature_matrix
        rr_feature_matrix = dp.rr_feature_matrix
        print 'signal length', len(signal)
        qrs_feature_matrix = np.expand_dims(qrs_feature_matrix, 1)
        rr_feature_matrix = np.expand_dims(rr_feature_matrix, 1)
        morph = get_r_peaks(qrs_feature_matrix, rr_feature_matrix)
        assert len(morph) == len(dp.original_r_peaks)
        morph = np.argmax(morph, axis=1)
        r_peaks = map(lambda (ind, ann): ind, dp.original_r_peaks)
        morph = map(lambda org_morph: annotation_dict[org_morph+1], morph)
        annot_list = zip(r_peaks, morph)

        #rr = np.diff(map(lambda (k, v): k,  annot_list))
        print 'saving annot file'
        file_path = os.path.join(db_path, record)
        wrann(annot_list, file_path + '.tan')
        print timeit.default_timer() - total_time

        os.chdir(db_path)
        call(['bxb', '-r', record, '-a', 'atr', 'tan', '-f', '0', '-L', 'result.bxb', '-'])
        call(['rxr', '-r', record, '-a', 'atr', 'tan', '-f', '0', '-L', 'result.rxr', 's_result.rxr'])
        os.chdir(base_dir)




if __name__ == '__main__':
    #cProfile.run('recognize_signal()')
    recognize_signal()
