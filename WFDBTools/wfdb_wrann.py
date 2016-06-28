__author__ = 'MW'
import numpy as np
from wfdb import rdann


def wrann(ann_list, path):
    file = open(path, 'wb')
    prev_sample_number = 0
    morph_skip = np.uint16(59)
    for annot in ann_list:
        delta = _delta_count(annot[0], prev_sample_number)
        if delta > 1024:
            prev_sample_number = annot[0]
            _write(file, 0, morph_skip)
            delta_H = np.uint16(delta) >> 16
            delta_H_L = delta_H >> 8
            delta_H_H = delta_H << 8
            delta_H = np.uint16(delta_H_H+delta_H_L)
            file.write(delta_H)
            delta_L = np.uint16(delta)
            delta_L_L = delta_L >> 8
            delta_L_H = delta_L << 8
            delta_L = np.uint16(delta_L_H+delta_L_L)
            file.write(np.uint16(delta))
            _write(file, 0, annot[1])
        else:

            prev_sample_number = annot[0]
            _write(file, delta, annot[1])
    file.write(np.uint16(0))
    file.close()


def _write(file_handle, delta, morf):
    temp = morf << 10
    sample = np.uint16(temp + delta)
    file_handle.write(sample)


def _write_extended(file_handle, delta, morf, morph_skip):
    _write(file_handle, delta, morf)
    file_handle.write(np.uint16(morph_skip << 16))


def _delta_count(sample_number, prev_sample_number):
    return sample_number - prev_sample_number


if __name__ == '__main__':
    #annots = rdann('C:\\Users\\user\\data\\mitdb\\100', 'atr', types=[1, 5, 6, 7, 8, 9, 10, 11])
    #annots = map(lambda annot: (int(annot[0]), int(annot[-1])), annots)
    #wrann(annots, 'C:\\Users\\user\\data\\mitdb\\100.oo')

    wrann([(100, 1), (500, 5), (5000, 5)], 'C:\\Users\\user\\data\\mitdb\\100.oo')
