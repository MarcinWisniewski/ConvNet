__author__ = 'MW'

import pickle as cPickle
import timeit

import numpy as np
import theano.tensor as T
import theano
import os
import matplotlib.pyplot as plt

from CNN.conv_network import CNN
from Readers.ecg_provider import DataProvider
from WFDBTools.wfdb_wrann import wrann
import cProfile


# dict from class to wfdb code
annotation_dict = {0: 0, 1: 1, 2: 5, 3: 9}
db_path = '/home/marcin/data/mitdb'

files = os.listdir(db_path)
files = [record.split('.')[0] for record in files if record.split('.')[-1] == 'dat']


def recognize_signal():
    x = T.matrix('x', dtype=theano.config.floatX)
    n_kerns = [10, 15, 20]
    batch_size = 1
    rng = np.random.RandomState(23455)
    f = open('model.bin', 'rb')
    cn_net = CNN(rng, x, n_kerns, batch_size)
    cn_net.__setstate__(cPickle.load(f))
    f.close()
    window = 1024
    for record in files:
        print '...analysing record', record
        total_time = timeit.default_timer()
        file_path = os.path.join(db_path, record)
        dp = DataProvider(file_path, split_factor=100, window=window)
        dp.prepare_signal()
        signal = dp.signal
        #reference_annots = dp.annots
        print 'signal length', len(signal)
        ppm = theano.function(inputs=[cn_net.layer0_input], outputs=cn_net.layer4.y_pred)
        start_position = 0
        end_position = len(signal)
        p = np.zeros((end_position - start_position)+513)

        start_time = timeit.default_timer()
        step = 8
        annot_list = []
        qrs_detected = False
        skip_cntr = 8
        for i in xrange(start_position, end_position, step):
            temp_sig = signal[i:i+window]
            if i % 10000 == 0:
                print 'index: ', i
                print timeit.default_timer() - start_time
                start_time = timeit.default_timer()
            if len(temp_sig) == window:
                single_test_data = np.array(temp_sig, ndmin=4)
                single_test_data = (single_test_data-single_test_data.mean())/np.abs(single_test_data.max())
                result = ppm(single_test_data)
                p[i+window/2] = result[0]
                hist, bounds = np.histogram(p[i-2*step:i:step], bins=[0, 1, 2, 3, 4], density=True)
                if hist[0] < 0.5 and not qrs_detected:
                    skip_cntr = 2
                    qrs_detected = True
                    sample_index = i-2*step
                    annotation = annotation_dict[int(hist.argmax())]
                    annot_list.append((sample_index, annotation))
                elif skip_cntr > 0:
                    skip_cntr -= 1
                else:
                    qrs_detected = False
                    skip_cntr = 0

        print 'saving annot file'
        wrann(annot_list, file_path +'.tan')
        print timeit.default_timer() - total_time


if __name__ == '__main__':
    #cProfile.run('recognize_signal()')
    recognize_signal()
