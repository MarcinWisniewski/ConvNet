__author__ = 'MW'

import os
import numpy as np
import time
import theano
from WFDBTools.wfdb import rdann, rdsamp
from scipy.signal import resample
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

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
annot_dict = {1: 0,                          #N
              5: 1, 6: 1, 10: 1,             #V, F, E
              7: 2, 8: 2, 9: 2, 11: 2, 4: 2, 35: 2} #J, A, S, j, a, n

INDEX_OF_BEAT_ANNOTS = range(1, 13) + [25, 34, 35, 38]


class DataProvider(object):
    def __init__(self, data_base_path, split_factor, window=1024, step=128, start=0, stop=-1,
                 channels_to_analyse=None, number_of_channel_to_analyse=None, seed=2222222):
        assert isinstance(data_base_path, str), 'wrong path'
        np.random.seed(seed)
        self.data_base_path = data_base_path
        self.window = window
        self.step = step
        self.start = start
        self.stop = stop
        self.channels_to_analyse = channels_to_analyse
        self.number_of_channel_to_analyse = number_of_channel_to_analyse
        self.qrs_feature_matrix = []
        self.class_matrix = []
        self.testing_split_factor = split_factor
        self.validation_split_factor = self.testing_split_factor + (100 - self.testing_split_factor) / 2

    def get_training_set(self):
        return self.qrs_feature_matrix[:self.testing_start_index], \
               self.rr_feature_matrix[:self.testing_start_index],\
               self.class_matrix[:self.testing_start_index]

    def get_testing_set(self):
        return self.qrs_feature_matrix[self.testing_start_index:self.validation_start_index], \
               self.rr_feature_matrix[self.testing_start_index:self.validation_start_index], \
               self.class_matrix[self.testing_start_index:self.validation_start_index]

    def get_validate_set(self):
        return self.qrs_feature_matrix[self.validation_start_index:], \
               self.rr_feature_matrix[self.validation_start_index:], \
               self.class_matrix[self.validation_start_index:]

    def get_data(self):
        return zip(self.qrs_feature_matrix, self.rr_feature_matrix), self.class_matrix

    def prepare_signal(self, record, channel, multiply_cls=True):
        record_path = os.path.join(self.data_base_path, record)
        timer_start = time.time()
        self._read_signal(record_path)
        self._read_r_peaks(record_path)
        if self.signal_info['samp_freq'] == 257:
            self._resample_data()

        self._organize_data(channel)
        if multiply_cls:
            self._multiplicate_classes(5, 40)
        timer_stop = time.time()
        self.testing_start_index = int(len(self.qrs_feature_matrix) * (float(self.testing_split_factor) / 100))
        self.validation_start_index = int(len(self.qrs_feature_matrix) *
                                          (float(self.validation_split_factor) / 100))
        print timer_stop - timer_start

    def _resample_data(self):
        self.signal = resample(self.signal, len(self.signal[0]) / 257.0 * 360, axis=1)
        self.r_peaks = map(lambda (r_index, morph): (int(r_index / 257.0 * 360), morph), self.r_peaks)
        self.signal_info['samp_freq'] = 360

    def _read_r_peaks(self, record_path):
        annots = rdann(record_path, 'atr', types=INDEX_OF_BEAT_ANNOTS, start=self.start, end=self.stop)
        self.r_peaks = map(lambda annot: (int(annot[0]+np.random.randint(low=-5,
                                                                         high=5,
                                                                         size=1)),
                                          int(annot[-1])),
                           annots)

        self.original_r_peaks = self.r_peaks[:]

    def _read_signal(self, record_path):
        self.signal = rdsamp(record_path, end=self.stop)
        self.signal_info = self.signal[1]
        self.signal = np.transpose(self.signal[0])[2:]

    def _organize_data(self, channel):
        self.class_matrix = []
        self.qrs_feature_matrix = []
        self.rr_feature_matrix = []
        rrs = map(lambda annot: int(annot[0]), self.r_peaks)
        rrs = np.diff(np.insert(rrs, 0, 0))

        for r_peak, morph in self.r_peaks:
            input_vector = np.zeros((1, self.window), dtype=theano.config.floatX)
            if r_peak-self.window/2 < 0:
                normalized_signal = self._normalyse(self.signal[channel][0:r_peak+(self.window/2)])
            else:
                normalized_signal = self._normalyse(self.signal[channel][r_peak-(self.window/2):r_peak+(self.window/2)])

            input_vector[0][0:len(normalized_signal)] = normalized_signal
            self.qrs_feature_matrix.append(input_vector)
            if morph in annot_dict.keys():
                self.class_matrix.append(annot_dict[morph])
            else:
                self.class_matrix.append(0)

        L_margin = 0
        R_margin = 16
        for i in xrange(len(rrs)):
            rr_chunk = np.zeros((1, 32))
            if i < 16:
                rr_chunk[0][16-L_margin:] = rrs[i-L_margin: i+16]
                L_margin += 1
            elif i >= len(rrs) - 16:
                rr_chunk[0][:16+R_margin] = rrs[i-16: i+R_margin]
                R_margin -= 1
            else:
                rr_chunk[0] = rrs[i-16: i+16]
            rr_chunk = np.asarray(rr_chunk, dtype=theano.config.floatX)
            self.rr_feature_matrix.append(self._normalyse_rr(rr_chunk))

    def _multiplicate_classes(self, v_multiplier, s_multiplier):
        for i in xrange(len(self.class_matrix)):
            if self.class_matrix[i] == 1:
                for multiplier in xrange(v_multiplier):
                    self.add_multiplied_elements(i)
            elif self.class_matrix[i] == 2:
                for multiplier in xrange(s_multiplier):
                    self.add_multiplied_elements(i)

    def add_multiplied_elements(self, i):
        self.class_matrix.append(self.class_matrix[i])
        self.qrs_feature_matrix.append(self.qrs_feature_matrix[i] *
                                       np.random.uniform(low=0.6,
                                                         high=1.2,
                                                         size=1))
        self.rr_feature_matrix.append(self.rr_feature_matrix[i] *
                                      np.random.uniform(low=0.6,
                                                        high=1.2,
                                                        size=1) +
                                      np.random.uniform(low=-0.05,
                                                        high=0.05,
                                                        size=len(self.rr_feature_matrix[i])))

    @staticmethod
    def _normalyse(frame):
        frame_copy = np.copy(frame)
        frame_copy -= frame_copy.min()
        return frame_copy

    def _normalyse_rr(self, frame):
        frame_copy = np.copy(frame)
        frame_copy /= float(self.signal_info['samp_freq'])
        return frame_copy

if __name__ == '__main__':
    ecg = DataProvider('/home/marcin/data/', 140, 1024)
    ecg.prepare_signal()
