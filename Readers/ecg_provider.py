__author__ = 'MW'

import os
import numpy as np
import time
import theano
from WFDBTools.wfdb import rdann, rdsamp
from scipy.signal import resample
import matplotlib.pyplot as plt


#1 : 'NORMAL',	# normal beat */
#4 : 'ABERR',	# aberrated atrial premature beat */
#5 : 'PVC',	# premature ventricular contraction */
#6 : 'FUSION',	# fusion of ventricular and normal beat */
#7 : 'NPC',	# nodal (junctional) premature beat */
#8 : 'APC',	# atrial premature contraction */
#9 : 'SVPB',	# premature or ectopic supraventricular beat */
#10 : 'VESC',	# ventricular escape beat */
#11 : 'NESC',	# nodal (junctional) escape beat */
#12 : 'PACE',	# paced beat */
annot_dict = {1: 1,                          #N
              5: 2, 6: 2, 10: 2,             #V, F, E
              7: 3, 8: 3, 9: 3, 11: 3, 4: 3} #J, A, S, j, a,

INDEX_OF_BEAT_ANNOTS = range(1, 13) + [25, 34, 35, 38]


class DataProvider(object):
    def __init__(self, data_base_path, split_factor, window=1024, step=128, start=0, stop=-1,
                 channels_to_analyse=None, number_of_channel_to_analyse=None):
        assert isinstance(data_base_path, str), 'wrong path'
        np.random.seed(2222222)
        self.data_base_path = data_base_path
        self.window = window
        self.step = step
        self.start = start
        self.stop = stop
        self.channels_to_analyse = channels_to_analyse
        self.number_of_channel_to_analyse = number_of_channel_to_analyse
        self.feature_matrix = []
        self.class_matrix = []
        self.testing_split_factor = split_factor
        self.validation_split_factor = self.testing_split_factor + (100 - self.testing_split_factor) / 2

    def get_training_set(self):
        return self.feature_matrix[:self.testing_start_index], self.class_matrix[:self.testing_start_index]

    def get_testing_set(self):
        return self.feature_matrix[self.testing_start_index:self.validation_start_index], \
               self.class_matrix[self.testing_start_index:self.validation_start_index]

    def get_validate_set(self):
        return self.feature_matrix[self.validation_start_index:], \
               self.class_matrix[self.validation_start_index:]

    def prepare_signal(self, record):
        record_path = os.path.join(self.data_base_path, record)
        timer_start = time.time()
        self._read_signal(record_path)
        self._read_r_peaks(record_path)
        if self.signal_info['samp_freq'] == 257:
            self._resample_data()
        self._organize_data()
        timer_stop = time.time()
        self.testing_start_index = int(len(self.feature_matrix) * (float(self.testing_split_factor) / 100))
        self.validation_start_index = int(len(self.feature_matrix) *
                                          (float(self.validation_split_factor) / 100))
        print timer_stop - timer_start

    def _resample_data(self):
        self.signal = resample(self.signal, len(self.signal[0]) / 257.0 * 360, axis=1)
        self.r_peaks = (self.r_peaks / 257.0 * 360)
        self.r_peaks = map(lambda r: int(r), self.r_peaks)

    def _read_r_peaks(self, record_path):
        annots = rdann(record_path, 'atr', types=INDEX_OF_BEAT_ANNOTS, start=self.start, end=self.stop)
        annots = map(lambda annot: (int(annot[0]), int(annot[-1])), annots)
        self.r_peaks = map(lambda annot: int(annot[0]), annots)
        self.r_peaks = np.asarray(self.r_peaks, dtype=('i4'))

    def _read_signal(self, record_path):
        self.signal = rdsamp(record_path, end=self.stop)
        self.signal_info = self.signal[1]
        self.signal = np.transpose(self.signal[0])[2:]

    def _organize_data(self):
        self.class_matrix = []
        self.feature_matrix = []
        channels, signal_length = self.signal.shape
        max_channels_in_input_vector = min(channels, 3)
        for i in xrange(0, signal_length-self.window, self.step):
            if self.channels_to_analyse is None or self.number_of_channel_to_analyse is None:
                number_of_channel_to_analyse = np.random.randint(1, max_channels_in_input_vector + 1)
                channels_to_analyse = np.random.randint(0, channels, number_of_channel_to_analyse)
            else:
                assert isinstance(self.channels_to_analyse, type([])), 'channel_to_analyse should be a list'
                assert self.number_of_channel_to_analyse <= max_channels_in_input_vector, 'to many channels to analyse'
                assert len(self.channels_to_analyse) <= max_channels_in_input_vector, 'chosen channel exceeds dimension'
                number_of_channel_to_analyse = self.number_of_channel_to_analyse
                channels_to_analyse = self.channels_to_analyse

            input_vector = np.zeros((3, self.window), dtype=theano.config.floatX)
            for channel in xrange(number_of_channel_to_analyse):
                frame = self._normalyse(self.signal[channels_to_analyse[channel], i:i+self.window])
                input_vector[channel] = frame

            r_peaks_in_frame = next((r_peak for r_peak in self.r_peaks if i < r_peak < i + self.window), 0)
            if r_peaks_in_frame != 0:
                r_peaks_in_frame = (r_peaks_in_frame - i) / float(self.window)

            if self.channels_to_analyse is None or self.number_of_channel_to_analyse is None:
                np.random.shuffle(input_vector)
            self.feature_matrix.append(input_vector)
            self.class_matrix.append(r_peaks_in_frame)


    @staticmethod
    def _normalyse(frame):
        frame_copy = np.copy(frame)
        frame_copy -= frame_copy.min()
        frame_copy /= frame_copy.max()
        return frame_copy


if __name__ == '__main__':
    ecg = DataProvider('/home/marcin/data/', 140, 1024)
    ecg.prepare_signal()
