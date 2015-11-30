__author__ = 'MW'

import random as rn
import numpy as np
import timeit
from WFDBTools.wfdb import rdann, rdsamp
from scipy.signal import sosfilt
#from scipy import signal as _signal
import matplotlib.pyplot as plt


SOS = [[1,	-2,	1,	1,	-1.99712697698516,	0.997192422553049],
       [1,	-2,	1,	1,	-1.99187190140264,	0.991937174762453],
       [1,	-2,	1,	1,	-1.98760834267259,	0.987673476316188],
       [1,	-2,	1,	1,	-1.98483533312596,	0.984900375898430],
       [1,	-1,	0,	1,	-0.991937043188383,	0]]


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
    def __init__(self, path, split_factor, window=1024, start=0, stop=-1):
        assert isinstance(path, str), 'wrong path'
        rn.seed(2121212)
        self.path = path
        self.WIN = window
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
        signal = rdsamp(self.path, end=self.stop)
        self.signal = np.asarray(map(lambda sample: sample[2], signal[0]))
        self.signal = sosfilt(SOS, self.signal)
        annots = rdann(self.path, 'atr', types=annot_dict.keys(), start=self.start, end=self.stop)
        annots = map(lambda annot: (int(annot[0]), int(annot[-1])), annots)
        self.annots = np.asarray(annots, dtype=('i4, i4'))
        all_annots = rdann(self.path, 'atr', types=range(1, 13), start=self.start, end=self.stop)
        all_annots = map(lambda annot: (int(annot[0]), int(annot[-1])), all_annots)
        self.all_annots = np.asarray(all_annots, dtype=('i4, i4'))
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
        CLASS_WIN = 16
        for annot in self.annots:
            if annot[0]-self.WIN/2 > 0 and annot[0]+self.WIN/2 < signal_length:
                for index in xrange(annot[0]-CLASS_WIN/2, annot[0]+CLASS_WIN/2):
                    frame = np.copy(self.signal[index-self.WIN/2:index+self.WIN/2])
                    if len(frame) == self.WIN:
                        frame = normalyse(frame)
                        self.inputMatrix.append(frame)
                        self.classMatrix.append(annot_dict[annot[-1]])
                        #self.classMatrix.append(1)

        for i in range(len(self.all_annots)-1):
            if self.all_annots[i][0]+CLASS_WIN/2-self.WIN/2 > 0 and \
                                            self.all_annots[i+1][0]-CLASS_WIN/2+self.WIN/2 < signal_length:
                rr_distance = self.all_annots[i+1][0] - self.all_annots[i][0] - CLASS_WIN
                indexes_of_isoline = range(self.all_annots[i][0]+CLASS_WIN/2,
                                           self.all_annots[i][0]+CLASS_WIN/2 + rr_distance)
                rn.shuffle(indexes_of_isoline)
                for index in xrange(CLASS_WIN*2):
                    frame = np.copy(self.signal[indexes_of_isoline[index]-self.WIN/2:
                    indexes_of_isoline[index]+self.WIN/2])
                    if len(frame) == self.WIN:
                        frame = normalyse(frame)
                        #plt.plot(frame)
                        self.inputMatrix.append(frame)
                        self.classMatrix.append(0)

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
        return self.inputMatrix[self.divide_index_test_valid_data_set:], self.classMatrix[self.divide_index_test_valid_data_set:]


if __name__ == '__main__':
    ecg = DataProvider('~/ubuntu/data/mitdb/100', 100, 1024)
    ecg.prepare_signal()
