__author__ = 'Marcin'
import lasagne


class CNN(object):
    def __init__(self, input, n_kerns, batch_size):

        assert len(n_kerns) == 5
        self.input_signal_length = 1024
        self.layer0_input = input
        self.layer0_input = input.reshape((batch_size, 1, self.input_signal_length))
        self.network = lasagne.layers.InputLayer(shape=(None, 1, self.input_signal_length),
                                                 input_var=self.layer0_input)

        self.network = lasagne.layers.Conv1DLayer(incoming=self.network, num_filters=n_kerns[0],
                                                  filter_size=71, pad='same',
                                                  nonlinearity=lasagne.nonlinearities.identity)
        self.network = lasagne.layers.MaxPool1DLayer(self.network, pool_size=2)

        self.network = lasagne.layers.Conv1DLayer(incoming=self.network, num_filters=n_kerns[1],
                                                  filter_size=71, pad='same',
                                                  nonlinearity=lasagne.nonlinearities.identity)
        self.network = lasagne.layers.MaxPool1DLayer(self.network, pool_size=2)

        self.network = lasagne.layers.Conv1DLayer(incoming=self.network, num_filters=n_kerns[2],
                                                  filter_size=35, pad='same',
                                                  nonlinearity=lasagne.nonlinearities.identity)
        self.network = lasagne.layers.MaxPool1DLayer(self.network, pool_size=2)

        self.network = lasagne.layers.Conv1DLayer(incoming=self.network, num_filters=n_kerns[3],
                                                  filter_size=35, pad='same',
                                                  nonlinearity=lasagne.nonlinearities.identity)

        self.network = lasagne.layers.Conv1DLayer(incoming=self.network, num_filters=n_kerns[4],
                                                  filter_size=35, pad='same',
                                                  nonlinearity=lasagne.nonlinearities.identity)

        self.network = lasagne.layers.DenseLayer(lasagne.layers.dropout(self.network, p=.5),
                                                 num_units=128, nonlinearity=lasagne.nonlinearities.rectify)
        self.network = lasagne.layers.DenseLayer(lasagne.layers.dropout(self.network, p=.5),
                                                 num_units=128, nonlinearity=lasagne.nonlinearities.rectify)

        self.network = lasagne.layers.DenseLayer(self.network, num_units=10,
                                                 nonlinearity=lasagne.nonlinearities.rectify)

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):

    def __getstate__(self):
        return lasagne.layers.get_all_param_values(self.network)

    def __setstate__(self, weights):
        lasagne.layers.set_all_param_values(self.network, weights)
