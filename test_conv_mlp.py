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
    n_kerns = [10, 10, 10]
    batch_size = 1
    rng = np.random.RandomState(23455)
    f = open('model.bin', 'rb')
    cn_net = CNN(rng, x, n_kerns, batch_size)
    cn_net.__setstate__(cPickle.load(f))
    f.close()

    dp = DataProvider('C:\\Users\\user\\data\\MIT\\100', 100, 1024)
    dp.prepare_signal()
    train_small_set = dp.getTrainingSet()

    test_data = train_small_set[0]
    N = len(test_data)
    test_data = theano.shared(np.asarray(test_data, dtype=theano.config.floatX))
    # just zeroes
    test_labels = T.cast(theano.shared(np.asarray(np.zeros(batch_size), dtype=theano.config.floatX)), 'int32')

    ppm = theano.function([index], cn_net.layer4.pred_probs(),
        givens={
            x: test_data[index * batch_size: (index + 1) * batch_size],
            y: test_labels
        }, on_unused_input='warn')

    #ppm = theano.function(inputs=[cn_net.layer0_input], outputs=cn_net.layer4.pred_probs())
    # p : predictions, we need to take argmax, p is 3-dim: (# loop iterations x batch_size x 2)
    p = [ppm(ii) for ii in range(N // batch_size)]

    p = np.array(p)
    p = p.reshape((N, 6))
    print (p)
    p = np.argmax(p, axis=1)
    p = p.astype(int)

    plt.plot(p*100)
    plt.plot(dp.signal)
    plt.show()
    print 1


if __name__ == '__main__':
    recognize_signal()
