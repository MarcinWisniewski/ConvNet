__author__ = 'MW'

import pickle as cPickle
import timeit

import numpy as np
import theano.tensor as T
import theano
import lasagne
import os
from subprocess import call
from CNN.conv_network import CNN
from Readers.ecg_provider import DataProvider
from WFDBTools.wfdb_wrann import wrann
import cProfile
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
qrs_n_kerns=(50, 65, 30, 32, 8)
rr_n_kerns=(45, 64, 50, 32, 8)
p2p_n_kerns=(32, 8)

# dict from class to wfdb code
annotation_dict = {0: 0, 1: 1, 2: 5, 3: 9}
db_path = os.path.expanduser('~/data/mitdb/')

record = '100'
n_index = 190
s_index = 230
v_index = 1906


def recognize_signal():
    if 'mitdb' in db_path:
        channel = 0
    else:
        channel = 1
    base_dir = os.getcwd()
    x_qrs = T.tensor4('x_qrs', dtype=theano.config.floatX)    # the data is presented as qrs normalized to [0-1]
    x_rr = T.tensor4('x_rr', dtype=theano.config.floatX)    # the data is presented as rrs normalized to by fs
    x_p2p = T.tensor4('x_rr', dtype=theano.config.floatX)    # the data is presented as p2p

    batch_size = 128
    seed = 23455
    f = open('qrs_model.bin', 'rb')
    cn_net = CNN(x_qrs, x_rr, x_p2p, qrs_n_kerns, rr_n_kerns, p2p_n_kerns, batch_size)
    cn_net.__setstate__(cPickle.load(f))
    f.close()
    qrs_cnn_out = lasagne.layers.get_output(cn_net.cnn_qrs_net, deterministic=True)
    rr_cnn_out = lasagne.layers.get_output(cn_net.cnn_rr_net, deterministic=True)
    p2p_cnn_out = lasagne.layers.get_output(cn_net.cnn_p2p_net, deterministic=True)
    nn_out = lasagne.layers.get_output(cn_net.mlp_net, deterministic=True)
    dp = DataProvider(db_path, split_factor=100, window=256,
                      start=0, stop=-1, seed=seed)
    get_nn_out = theano.function([x_qrs, x_rr, x_p2p], nn_out)
    get_qrs_cnn_out = theano.function([x_qrs, x_rr, x_p2p], qrs_cnn_out, on_unused_input='ignore')
    get_rr_cnn_out = theano.function([x_qrs, x_rr, x_p2p], rr_cnn_out, on_unused_input='ignore')
    get_p2p_cnn_out = theano.function([x_qrs, x_rr, x_p2p], p2p_cnn_out, on_unused_input='ignore')

    print '...analysing record', record
    total_time = timeit.default_timer()
    dp.prepare_signal(record, channel=channel, multiply_cls=False)
    signal = dp.signal
    qrs_feature_matrix = dp.qrs_feature_matrix
    rr_feature_matrix = dp.rr_feature_matrix
    p2p_feature_matrix = dp.p2p_feature_matrix
    print 'signal length', len(signal)
    qrs_feature_matrix = np.expand_dims(qrs_feature_matrix, 1)
    rr_feature_matrix = np.expand_dims(rr_feature_matrix, 1)
    p2p_feature_matrix = np.expand_dims(p2p_feature_matrix, 1)
    nn_out = get_nn_out(qrs_feature_matrix, rr_feature_matrix, p2p_feature_matrix)
    qrs_cnn_out = get_qrs_cnn_out(qrs_feature_matrix, rr_feature_matrix, p2p_feature_matrix)
    rr_cnn_out = get_rr_cnn_out(qrs_feature_matrix, rr_feature_matrix, p2p_feature_matrix)
    p2p_cnn_out = get_p2p_cnn_out(qrs_feature_matrix, rr_feature_matrix, p2p_feature_matrix)

    morph = np.argmax(nn_out, axis=1)

    plt.figure(1)
    plt.subplot('311')
    plt.imshow(qrs_cnn_out[n_index].reshape(8, 32))
    plt.colorbar()

    plt.subplot('312')
    plt.imshow(qrs_cnn_out[s_index].reshape(8, 32))
    plt.colorbar()

    plt.subplot('313')
    plt.imshow(qrs_cnn_out[v_index].reshape(8, 32))
    plt.colorbar()

    plt.figure(2)
    plt.subplot('311')
    plt.imshow(rr_cnn_out[n_index].reshape(8, 16))
    plt.colorbar()

    plt.subplot('312')
    plt.imshow(rr_cnn_out[s_index].reshape(8, 16))
    plt.colorbar()

    plt.subplot('313')
    plt.imshow(rr_cnn_out[v_index].reshape(8, 16))
    plt.colorbar()

    plt.figure(3)
    plt.subplot('311')
    plt.imshow(p2p_cnn_out[n_index].reshape(8, 32))
    plt.colorbar()

    plt.subplot('312')
    plt.imshow(p2p_cnn_out[s_index].reshape(8, 32))
    plt.colorbar()

    plt.subplot('313')
    plt.imshow(p2p_cnn_out[v_index].reshape(8, 32))
    plt.colorbar()


    plt.show()

if __name__ == '__main__':
    #cProfile.run('recognize_signal()')
    recognize_signal()
