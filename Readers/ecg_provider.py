__author__ = 'MW'

import random as rn
import numpy as np
import timeit
from WFDBTools.wfdb import rdann, rdsamp
from scipy.signal import sosfilt
#from scipy import signal as _signal
import matplotlib.pyplot as plt




annot_dict = {1: 1,                          #N
              5: 2, 6: 2, 10: 2,             #V, F, E
              7: 3, 8: 3, 9: 3, 11: 3, 4: 3} #J, A, S, j, a,

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
        rn.seed(2121212)
        self.path = path
        self.WIN = window
        self.step = step
        self.start = start
        self.stop = stop
        self.inputMatrix = []
        self.classMatrix = []
        self.percentageSplit = split_factor
        self.percentageSplitTestValid = self.percentageSplit+(100-self.percentageSplit)/2

    def __del__(self):
        del self.inputMatrix[:]
        del self.classMatrix[:]
        print 'object cleaned'

    def prepare_signal(self):
        timer_start = timeit.default_timer()
        self.signal = rdsamp(self.path, end=self.stop)
        self.signal = np.asarray(map(lambda sample: sample[2], self.signal[0]))
        annots = rdann(self.path, 'atr', types=range(1, 13) + [34, 35, 38], start=self.start, end=self.stop)
        annots = map(lambda annot: (int(annot[0]), int(annot[-1])), annots)
        #self.annots = np.asarray(annots, dtype=('i4, i4'))
        self.r_peaks = map(lambda annot: int(annot[0]), annots)
        self.r_peaks = np.asarray(self.r_peaks, dtype=('i4'))

        self.organize_data()
        timer_stop = timeit.default_timer()
        print timer_stop - timer_start

        self.divide_index = int(len(self.inputMatrix) * (float(self.percentageSplit) / 100))
        self.divide_index_test_valid_data_set = int(len(self.inputMatrix) *
                                                    (float(self.percentageSplitTestValid) / 100))

    def organize_data(self):
        def normalyse(frame):
            frame -= frame.mean()
            frame /= np.abs(frame.max())
            return frame
        signal_length = len(self.signal)
        for i in xrange(0, signal_length-self.WIN, self.step):
            frame = normalyse(self.signal[i:i+self.WIN])
            r_peaks_in_frame = [r_peak for r_peak in self.r_peaks if r_peak > i and r_peak < i+self.WIN]
            r_peaks_in_frame = (r_peaks_in_frame-i*np.ones(len(r_peaks_in_frame)))/1024.0
            if len(r_peaks_in_frame) != 0:
                self.inputMatrix.append(frame)
                target = np.zeros(10)
                target[0:len(r_peaks_in_frame)] = r_peaks_in_frame
                self.classMatrix.append(target)

    def reshuffle_data(self):
        tempMtx = []
        tempCls = []
        indexes = range(len(self.inputMatrix))
        rn.shuffle(indexes)
        for i in indexes:
            tempMtx.append(self.inputMatrix[i])
            tempCls.append(self.classMatrix[i])
        self.classMatrix = tempCls[:]
        self.inputMatrix = tempMtx[:]

    def getTrainingSet(self):
        return self.inputMatrix[:self.divide_index], self.classMatrix[:self.divide_index]

    def getTestingSet(self):
        return self.inputMatrix[self.divide_index:self.divide_index_test_valid_data_set], \
               self.classMatrix[self.divide_index:self.divide_index_test_valid_data_set]

    def getValidateSet(self):
        return self.inputMatrix[self.divide_index_test_valid_data_set:], \
               self.classMatrix[self.divide_index_test_valid_data_set:]


if __name__ == '__main__':
    ecg = DataProvider('/home/marcin/data/mitdb/124', 140, 1024)
    ecg.prepare_signal()
