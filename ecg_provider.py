__author__ = 'MW'

import random as rn
import matplotlib.pyplot as plt
import numpy as np
import timeit
from wfdb import rdann, rdsamp, CODEDICT


annot_dict = {1: 1, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 2, 11: 3}


class DataProvider(object):
    def __init__(self, path, split_factor, window):
        assert isinstance(path, str), 'wrong path'
        rn.seed(2121212)
        self.path = path
        self.WIN = window
        self.inputMatrix = []
        self.classMatrix = []
        self.percentageSplit = split_factor
        self.percentageSplitTestValid = self.percentageSplit+(100-self.percentageSplit)/2

    def __del__(self):
        del self.inputMatrix[:]
        del self.classMatrix[:]
        print 'object cleaned'

    def prepare_signal(self):
        start = 100
        stop = 700
        self.data_path = self.path
        timer_start = timeit.default_timer()
        signal = rdsamp(self.path, start=start, end=stop)
        self.signal = np.asarray(map(lambda sample: sample[2], signal[0]))
        annots = rdann(self.path, 'atr', types=[1, 5, 6, 7, 8, 9, 10, 11], start=start, end=stop)
        annots = map(lambda annot: (int(annot[0]), int(annot[-1])), annots)
        self.annots = np.asarray(annots, dtype=('i4, i4'))
        self.organize_data()
        timer_stop = timeit.default_timer()
        print timer_stop - timer_start
        self.divideIndex = int(len(self.inputMatrix)*(float(self.percentageSplit)/100))
        self.divideIndexTestValid = int(len(self.inputMatrix)*(float(self.percentageSplitTestValid)/100))

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
                    #plt.plot(frame)
                    #plt.show()

        for annot_index in xrange(len(self.annots)-1):
            if self.annots[annot_index][0]-self.WIN/2 > 0 and \
                                    self.annots[annot_index+1][0]+self.WIN/2 < signal_length:
                rr = (self.annots[annot_index+1][0]-CLASS_WIN)-(self.annots[annot_index][0]+CLASS_WIN)
                if rr > CLASS_WIN:
                    indexes = range(rr)
                    rn.shuffle(indexes)
                    for index in indexes[:CLASS_WIN]:
                        reference_index = self.annots[annot_index][0]+CLASS_WIN + index
                        frame = np.copy(self.signal[reference_index - self.WIN/2 : reference_index + self.WIN/2])
                        if len(frame) == self.WIN:
                            frame = normalyse(frame)
                            self.inputMatrix.append(frame)
                            self.classMatrix.append(0)
                        #plt.plot(frame)
                        #plt.show()


    def reshuffleData(self):
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
        return self.inputMatrix[:self.divideIndex], self.classMatrix[:self.divideIndex]

    def getTestingSet(self):
        return self.inputMatrix[self.divideIndex:self.divideIndexTestValid], self.classMatrix[self.divideIndex:self.divideIndexTestValid]

    def getValidateSet(self):
        return self.inputMatrix[self.divideIndexTestValid:], self.classMatrix[self.divideIndexTestValid:]


if __name__ == '__main__':
    ecg = DataProvider('C:\\Users\\user\\data\\mitdb\\100', 100, 1024)
    ecg.prepare_signal()