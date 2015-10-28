__author__ = 'MW'


from conv_network import CNN
import pickle as cPickle
import numpy as np
import theano.tensor as T
import theano
from ecg_provider import DataProvider
import matplotlib.pyplot as plt

theano.config.exception_verbosity = 'high'


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

    dp = DataProvider('C:\\Users\\user\\data\\MIT\\106', 100, 1024)
    dp.prepare_signal()
    train_small_set = dp.getTrainingSet()
    signal = dp.signal
    test_data = train_small_set[0]
    N = len(test_data)
    ppm = theano.function(inputs=[cn_net.layer0_input], outputs=cn_net.layer4.y_pred)
    start_position = 0
    end_position = 100000
    p = np.zeros(start_position+512)
    p = list(p)
    for i in xrange(start_position, end_position, 2):
        single_test_data = np.array(signal[i:i+1024], ndmin=4)
        single_test_data -= single_test_data.mean()
        single_test_data /= np.abs(single_test_data.max())
        result = ppm(single_test_data)
        p.append(result[0])
        p.append(result[0])

    plt.plot(p)
    plt.plot((dp.signal-dp.signal.mean())/(dp.signal.max()*0.2))
    plt.show()
    print 1


if __name__ == '__main__':
    recognize_signal()
