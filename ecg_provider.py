__author__ = 'MW'

import random as rn
import matplotlib.pyplot as plt
import numpy as np
import timeit


annot_dict = {'N': 1, 'V': 2, 'S': 3, 'F': 4, 'A': 5}


class DataProvider(object):
    def __init__(self, path, split_factor, window):
        assert isinstance(path, str), 'wrong path'
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
        self.data_path = self.path + ".csv"
        self.annotation_path = self.path + '.ann'
        self._signal_reader = open(self.data_path)
        self.annot_reader = open(self.annotation_path)
        self.read_file()
        self.organize_data()
        self._signal_reader.close()
        self.annot_reader.close()
        self.divideIndex = int(len(self.inputMatrix)*(float(self.percentageSplit)/100))
        self.divideIndexTestValid = int(len(self.inputMatrix)*(float(self.percentageSplitTestValid)/100))

    def read_file(self):
        self._signal_reader.next()
        self._signal_reader.next()
        signal = []
        timer_start = timeit.default_timer()
        for line in self._signal_reader:
            line = line.split()
            signal.append(float(line[-2]))
        self.signal = np.asarray(signal)

        self.annot_reader.next()
        annots = []
        for line in self.annot_reader:
            line = line.split()
            if line[2] in annot_dict.keys():
                annots.append((int(line[1]), line[2], annot_dict[line[2]]))
        self.annots = np.asarray(annots, dtype=('i4, a3, i3'))
        timer_stop = timeit.default_timer()
        print timer_stop - timer_start

    def organize_data(self):
        MAX_WIN = 10
        index = 0
        while self.annots[index][0] < self.WIN/2:
            index += 1
        i_win = MAX_WIN
        for i in range(len(self.signal)-self.WIN):
            strip = self.signal[i:i+self.WIN]
            strip = (strip - strip.mean())/strip.max()
            self.inputMatrix.append(strip)
            if i + self.WIN/2 + MAX_WIN/2 == self.annots[index][0]:
                self.classMatrix.append(self.annots[index][2])
                i_win -= 1
                index += 1
            elif i_win < MAX_WIN and i_win > 0:
                self.classMatrix.append(self.annots[index-1][2])
                i_win -= 1
            else:
                if i % 10 == 0:
                    self.classMatrix.append(0)
                    i_win = MAX_WIN
                else:
                    self.inputMatrix.pop()

    def reshuffleData(self):
        rn.seed(2121212)
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
    ecg = DataProvider('C:\\Users\\user\\data\\15s')
