__author__ = 'MW'

import random as rn
import matplotlib.pyplot as plt
import numpy as np
import timeit


annot_dict = {'N': 1, 'V': 2, 'S': 3, 'F': 4, 'A': 5}


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
        self.data_path = self.path + ".csv"
        self.annotation_path = self.path + '.ann'
        self._signal_reader = open(self.data_path)
        self.annot_reader = open(self.annotation_path)
        self.read_file()
        timer_start = timeit.default_timer()
        self.organize_data()
        timer_stop = timeit.default_timer()
        print timer_stop - timer_start
        self._signal_reader.close()
        self.annot_reader.close()
        self.divideIndex = int(len(self.inputMatrix)*(float(self.percentageSplit)/100))
        self.divideIndexTestValid = int(len(self.inputMatrix)*(float(self.percentageSplitTestValid)/100))

    def read_file(self):
        signal = []
        for line in self._signal_reader:
            line = line.split()
            signal.append(float(line[-2]))
        self.signal = np.asarray(signal)

        self.annot_reader.next()
        annots = []
        # annot model: [sample_no, annot, annot_code]
        for line in self.annot_reader:
            line = line.split()
            if line[2] in annot_dict.keys():
                annots.append((int(line[1]), line[2], annot_dict[line[2]]))
        self.annots = np.asarray(annots, dtype=('i4, a3, i3'))

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
                        self.classMatrix.append(annot[-1])
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
    ecg = DataProvider('C:\\Users\\user\\data\\MIT\\100', 100, 1024)
    ecg.prepare_signal()