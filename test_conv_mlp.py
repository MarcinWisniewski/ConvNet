__author__ = 'MW'

import pickle as cPickle
import timeit

import numpy as np
import theano.tensor as T
import theano
import matplotlib.pyplot as plt

from CNN.conv_network import CNN
from Readers.ecg_provider import DataProvider
from WFDBTools.wfdb_wrann import wrann
import cProfile

theano.config.exception_verbosity = 'high'

# dict from class to wfdb code
annotation_dict = {0: 0, 1: 1, 2: 5, 3: 9}


def recognize_signal():
    x = T.matrix('x', dtype=theano.config.floatX)
    y = T.ivector('y')
    index = T.lscalar('index')
    n_kerns = [10, 20, 20]
    batch_size = 1
    rng = np.random.RandomState(23455)
    f = open('model_v3.bin', 'rb')
    cn_net = CNN(rng, x, n_kerns, batch_size)
    cn_net.__setstate__(cPickle.load(f))
    f.close()
    window = 1024
    file = 'C:\\Users\\user\\data\\mitdb\\222'
    dp = DataProvider(file, split_factor=100, window=window)
    dp.prepare_signal()
    signal = dp.signal
    print len(signal)
    ppm = theano.function(inputs=[cn_net.layer0_input], outputs=cn_net.layer4.y_pred)
    start_position = 0
    end_position = len(signal)
    p = np.zeros((end_position - start_position)+513)

    start_time = timeit.default_timer()
    step = 8
    for i in xrange(start_position, end_position, step):
        temp_sig = signal[i:i+window]
        if i % 1000 == 0:
            print i
            print timeit.default_timer() - start_time
            start_time = timeit.default_timer()
        if len(temp_sig) == window:
            single_test_data = np.array(temp_sig, ndmin=4)
            single_test_data = (single_test_data-single_test_data.mean())/np.abs(single_test_data.max())
            #single_test_data /= np.abs(single_test_data.max())
            result = ppm(single_test_data)
            p[i+window/2: i+window/2+step] = result[0]
            #p[i+window/2] = result[0]

    previous_no_qrs_probability = 1
    annot_list = []
    qrs_detected = False
    WIN = 16
    print 'Finding qrs positions'
    for i in xrange(len(p)-WIN):
        hist, bounds = np.histogram(p[i:i+WIN], bins=[0, 1, 2, 3, 4], density=True)
        if hist[0] < previous_no_qrs_probability and hist[0] < 0.25:
            qrs_detected = True
            sample_index = i+WIN/2
            annotation = annotation_dict[int(hist.argmax())]
        elif hist[0] < previous_no_qrs_probability and qrs_detected:
            annot_list.append((sample_index, annotation))
            qrs_detected = False
        previous_no_qrs_probability = hist[0]

    print 'saving annot file'
    wrann(annot_list, file +'.ta5')
    plt.plot((dp.signal-dp.signal.mean())/(dp.signal.max()*0.2))
    plt.plot(p, 'r')
    plt.show()
    print 1


if __name__ == '__main__':
    #cProfile.run('recognize_signal()')
    recognize_signal()
