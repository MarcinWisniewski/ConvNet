__author__ = 'MW'


from conv_network import CNN
import pickle as cPickle
import numpy as np
import theano.tensor as T
import theano
from ecg_provider import DataProvider
import matplotlib.pyplot as plt
from wfdb_wrann import wrann
import timeit

theano.config.exception_verbosity = 'high'

# dict from class to wfdb code
annotation_dict = {1: 1, 2: 5, 3: 9}
def recognize_signal():
    x = T.matrix('x', dtype=theano.config.floatX)
    y = T.ivector('y')
    index = T.lscalar('index')
    n_kerns = [10, 20, 20]
    batch_size = 1
    rng = np.random.RandomState(23455)
    f = open('model.bin', 'rb')
    cn_net = CNN(rng, x, n_kerns, batch_size)
    cn_net.__setstate__(cPickle.load(f))
    f.close()

    dp = DataProvider('C:\\Users\\user\\data\\mitdb\\100', 100, 1024)
    dp.prepare_signal()
    signal = dp.signal
    print len(signal)
    ppm = theano.function(inputs=[cn_net.layer0_input], outputs=cn_net.layer4.y_pred)
    start_position = 0
    end_position = 650000
    p = np.zeros((end_position - start_position)+513)

    start_time = timeit.default_timer()
    for i in xrange(start_position, end_position, 2):
        temp_sig = signal[i:i+1024]
        if i % 1000 == 0:
            print i
            print timeit.default_timer() - start_time
            start_time = timeit.default_timer()
        if len(temp_sig) == 1024:
            single_test_data = np.array(temp_sig, ndmin=4)
            single_test_data = (single_test_data-single_test_data.mean())/np.abs(single_test_data.max())
            #single_test_data /= np.abs(single_test_data.max())
            result = ppm(single_test_data)
            p[i+512] = result[0]
            p[i+1+512] = result[0]

    previous_no_qrs_probability = 1
    annot_list = []
    qrs_detected = False
    WIN = 32
    for i in xrange(len(p)-WIN):
        hist, baunds = np.histogram(p[i:i+WIN], bins=[0, 1, 2, 3, 4], density=True)
        if hist[0] < previous_no_qrs_probability and hist[0] < 0.25:
            qrs_detected = True
            sample_index = i+WIN/2
            annotation = annotation_dict[hist.argmax()]
        elif hist[0] < previous_no_qrs_probability and qrs_detected:
            annot_list.append((sample_index, annotation))
            qrs_detected = False

        previous_no_qrs_probability = hist[0]

    wrann(annot_list, 'C:\\Users\\user\\data\\mitdb\\100.ta')
    plt.plot(p, 'r')
    plt.plot((dp.signal-dp.signal.mean())/(dp.signal.max()*0.2))
    plt.show()
    print 1


if __name__ == '__main__':
    recognize_signal()
