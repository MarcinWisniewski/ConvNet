__author__ = 'Marcin'
import lasagne
from lasagne.layers import ConcatLayer


class CNN(object):
    def __init__(self, input_qrs, input_rr, input_p2p, qrs_n_kerns, rr_n_kerns, p2p_n_kerns, batch_size):

        assert len(qrs_n_kerns) == 5
        self.input_signal_length = 256
        self.input_rr_length = 32
        self.input_p2p_length = 32

        self.layer0_input_qrs = input_qrs
        self.layer0_input_rr = input_rr
        self.layer0_input_p2p = input_p2p

        self.cnn_qrs_net = lasagne.layers.InputLayer(shape=(None, 1, 1, self.input_signal_length),
                                                     input_var=self.layer0_input_qrs)

        self.cnn_qrs_net = lasagne.layers.Conv2DLayer(incoming=self.cnn_qrs_net, num_filters=qrs_n_kerns[0],
                                                      filter_size=(1, 15), pad='same',
                                                      W=lasagne.init.GlorotUniform(),
                                                      nonlinearity=lasagne.nonlinearities.rectify)
        self.cnn_qrs_net = lasagne.layers.MaxPool2DLayer(self.cnn_qrs_net, pool_size=(1, 2))

        self.cnn_qrs_net = lasagne.layers.Conv2DLayer(incoming=self.cnn_qrs_net, num_filters=qrs_n_kerns[1],
                                                      filter_size=(1, 7), pad='same',
                                                      W=lasagne.init.GlorotUniform(),
                                                      nonlinearity=lasagne.nonlinearities.rectify)
        self.cnn_qrs_net = lasagne.layers.MaxPool2DLayer(self.cnn_qrs_net, pool_size=(1, 2))

        self.cnn_qrs_net = lasagne.layers.Conv2DLayer(incoming=self.cnn_qrs_net, num_filters=qrs_n_kerns[2],
                                                      filter_size=(1, 3), pad='same',
                                                      W=lasagne.init.GlorotUniform(),
                                                      nonlinearity=lasagne.nonlinearities.rectify)
        self.cnn_qrs_net = lasagne.layers.MaxPool2DLayer(self.cnn_qrs_net, pool_size=(1, 2))

        self.cnn_qrs_net = lasagne.layers.Conv2DLayer(incoming=self.cnn_qrs_net, num_filters=qrs_n_kerns[3],
                                                      filter_size=(1, 3), pad='same',
                                                      W=lasagne.init.GlorotUniform(),
                                                      nonlinearity=lasagne.nonlinearities.rectify)

        self.cnn_qrs_net = lasagne.layers.Conv2DLayer(incoming=self.cnn_qrs_net, num_filters=qrs_n_kerns[4],
                                                      filter_size=(1, 3), pad='same',
                                                      W=lasagne.init.GlorotUniform(),
                                                      nonlinearity=lasagne.nonlinearities.rectify)

        ######################################################################################

        self.cnn_rr_net = lasagne.layers.InputLayer(shape=(None, 1, 1, self.input_rr_length),
                                                    input_var=self.layer0_input_rr)

        self.cnn_rr_net = lasagne.layers.Conv2DLayer(incoming=self.cnn_rr_net, num_filters=rr_n_kerns[0],
                                                     filter_size=(1, 5), pad='same',
                                                     W=lasagne.init.GlorotUniform(),
                                                     nonlinearity=lasagne.nonlinearities.rectify)

        self.cnn_rr_net = lasagne.layers.Conv2DLayer(incoming=self.cnn_rr_net, num_filters=rr_n_kerns[1],
                                                     filter_size=(1, 5), pad='same',
                                                     W=lasagne.init.GlorotUniform(),
                                                     nonlinearity=lasagne.nonlinearities.rectify)
        self.cnn_rr_net = lasagne.layers.MaxPool2DLayer(self.cnn_rr_net, pool_size=(1, 2))

        self.cnn_rr_net = lasagne.layers.Conv2DLayer(incoming=self.cnn_rr_net, num_filters=rr_n_kerns[2],
                                                     filter_size=(1, 3), pad='same',
                                                     W=lasagne.init.GlorotUniform(),
                                                     nonlinearity=lasagne.nonlinearities.rectify)

        self.cnn_rr_net = lasagne.layers.Conv2DLayer(incoming=self.cnn_rr_net, num_filters=rr_n_kerns[3],
                                                     filter_size=(1, 3), pad='same',
                                                     W=lasagne.init.GlorotUniform(),
                                                     nonlinearity=lasagne.nonlinearities.rectify)

        self.cnn_rr_net = lasagne.layers.Conv2DLayer(incoming=self.cnn_rr_net, num_filters=rr_n_kerns[4],
                                                     filter_size=(1, 3), pad='same',
                                                     W=lasagne.init.GlorotUniform(),
                                                     nonlinearity=lasagne.nonlinearities.rectify)

        ######################################################################################

        self.cnn_p2p_net = lasagne.layers.InputLayer(shape=(None, 1, 1, self.input_p2p_length),
                                                     input_var=self.layer0_input_p2p)

        self.cnn_p2p_net = lasagne.layers.Conv2DLayer(incoming=self.cnn_p2p_net, num_filters=p2p_n_kerns[0],
                                                      filter_size=(1, 3), pad='same',
                                                      W=lasagne.init.GlorotUniform(),
                                                      nonlinearity=lasagne.nonlinearities.rectify)

        self.cnn_p2p_net = lasagne.layers.Conv2DLayer(incoming=self.cnn_p2p_net, num_filters=p2p_n_kerns[1],
                                                      filter_size=(1, 3), pad='same',
                                                      W=lasagne.init.GlorotUniform(),
                                                      nonlinearity=lasagne.nonlinearities.rectify)

        self.cnn_p2p_net = lasagne.layers.MaxPool2DLayer(self.cnn_p2p_net, pool_size=(1, 2))

        self.cnn_p2p_net = lasagne.layers.Conv2DLayer(incoming=self.cnn_p2p_net, num_filters=p2p_n_kerns[2],
                                                      filter_size=(1, 3), pad='same',
                                                      W=lasagne.init.GlorotUniform(),
                                                      nonlinearity=lasagne.nonlinearities.rectify)

        self.cnn_p2p_net = lasagne.layers.Conv2DLayer(incoming=self.cnn_p2p_net, num_filters=p2p_n_kerns[3],
                                                      filter_size=(1, 3), pad='same',
                                                      W=lasagne.init.GlorotUniform(),
                                                      nonlinearity=lasagne.nonlinearities.rectify)

        qrs_rr_layer = ConcatLayer([self.cnn_qrs_net, self.cnn_rr_net, self.cnn_p2p_net], axis=-1)

        self.mlp_net = lasagne.layers.DenseLayer(lasagne.layers.dropout(qrs_rr_layer, p=.5),
                                                 num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
        self.mlp_net = lasagne.layers.DenseLayer(lasagne.layers.dropout(self.mlp_net, p=.5),
                                                 num_units=256, nonlinearity=lasagne.nonlinearities.rectify)

        self.mlp_net = lasagne.layers.DenseLayer(self.mlp_net, num_units=3,
                                                 nonlinearity=lasagne.nonlinearities.softmax)

    def __getstate__(self):
        return lasagne.layers.get_all_param_values(self.mlp_net)

    def __setstate__(self, weights):
        lasagne.layers.set_all_param_values(self.mlp_net, weights)
