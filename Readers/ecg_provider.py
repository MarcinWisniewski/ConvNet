__author__ = 'MW'

import random as rn
import numpy as np
import timeit
import theano
from WFDBTools.wfdb import rdann, rdsamp
from scipy.signal import resample
#from scipy import signal as _signal
import matplotlib.pyplot as plt




annot_dict = {1: 1,                          #N
              5: 2, 6: 2, 10: 2,             #V, F, E
              7: 3, 8: 3, 9: 3, 11: 3, 4: 3} #J, A, S, j, a,


INDEX_OF_BEAT_ANNOTS = range(1, 13) + [34, 35, 38]


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


class DataProvider(object):
    def __init__(self, path, split_factor, window=1024, step=128, start=0, stop=-1):
        assert isinstance(path, str), 'wrong path'
        np.random.seed(2222222)
        self.path = path
        self.WIN = window
        self.step = step
        self.start = start
        self.stop = stop
        self.inputMatrix = []
        self.classMatrix = []
        self.percentageSplit = split_factor
        self.percentageSplitTestValid = self.percentageSplit+(100-self.percentageSplit)/2

    def prepare_signal(self):
        timer_start = timeit.default_timer()
        self.signal = rdsamp(self.path, end=self.stop)
        self.signal_info = self.signal[1]
        self.signal = np.transpose(self.signal[0])[2:]
        annots = rdann(self.path, 'atr', types=INDEX_OF_BEAT_ANNOTS, start=self.start, end=self.stop)
        annots = map(lambda annot: (int(annot[0]), int(annot[-1])), annots)
        self.r_peaks = map(lambda annot: int(annot[0]), annots)
        self.r_peaks = np.asarray(self.r_peaks, dtype=('i4'))
        if self.signal_info['samp_freq'] == 257:
            self.signal = resample(self.signal, len(self.signal[0])/257.0*360, axis=1)
            self.r_peaks = (self.r_peaks/257.0*360)
            self.r_peaks = map(lambda r: int(r), self.r_peaks)
        self.organize_data()
        timer_stop = timeit.default_timer()
        print timer_stop - timer_start

        self.divide_index = int(len(self.inputMatrix) * (float(self.percentageSplit) / 100))
        self.divide_index_test_valid_data_set = int(len(self.inputMatrix) *
                                                    (float(self.percentageSplitTestValid) / 100))

    def organize_data(self):
        channels, signal_length = self.signal.shape
        max_channels_in_input_vector = min(channels, 3)
        for i in xrange(0, signal_length-self.WIN, self.step):
            number_of_signal_in_input = np.random.randint(1, max_channels_in_input_vector+1)
            channels_to_analyse = np.random.randint(0, channels, number_of_signal_in_input)
            input_vector = np.zeros((3, self.WIN), dtype=theano.config.floatX)
            for channel in xrange(number_of_signal_in_input):
                frame = self._normalyse(self.signal[channels_to_analyse[channel], i:i+self.WIN])
                input_vector[channel] = frame

            r_peaks_in_frame = [r_peak for r_peak in self.r_peaks if r_peak > i and r_peak < i+self.WIN]
            r_peaks_in_frame = (r_peaks_in_frame-i*np.ones(len(r_peaks_in_frame)))/1024.0
            if len(r_peaks_in_frame) != 0:
                np.random.shuffle(input_vector)
                self.inputMatrix.append(input_vector)
                target = np.zeros(10)
                target[0:len(r_peaks_in_frame)] = r_peaks_in_frame
                self.classMatrix.append(target)

    def reshuffle_data(self):
        if len(self.inputMatrix) != 0:
            input_output_tuple = zip(self.inputMatrix, self.classMatrix)
            np.random.shuffle(input_output_tuple)
            self.inputMatrix, self.classMatrix = zip(*input_output_tuple)


    def getTrainingSet(self):
        return self.inputMatrix[:self.divide_index], self.classMatrix[:self.divide_index]

    def getTestingSet(self):
        return self.inputMatrix[self.divide_index:self.divide_index_test_valid_data_set], \
               self.classMatrix[self.divide_index:self.divide_index_test_valid_data_set]

    def getValidateSet(self):
        return self.inputMatrix[self.divide_index_test_valid_data_set:], \
               self.classMatrix[self.divide_index_test_valid_data_set:]
    @staticmethod
    def _normalyse(frame):
        frame_copy = np.copy(frame)
        frame_copy -= frame_copy.mean()
        frame_copy /= np.abs(frame_copy).max()
        return frame_copy
if __name__ == '__main__':
    ecg = DataProvider('/home/marcin/data/', 140, 1024)
    ecg.prepare_signal()
