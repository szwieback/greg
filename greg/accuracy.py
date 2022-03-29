'''
Created on Dec 1, 2021

@author: szwieback
'''

import numpy as np

def _cdeviation(cest, ctrue=None):
    if ctrue is not None:
        cdev = cest * ctrue.conj()
    else:
        cdev = cest.copy()
    cdev /= np.abs(cdev)
    return cdev

def circular_accuracy(cest, ctrue=None, axis=0):
    cdev = _cdeviation(cest, ctrue=ctrue)
    acc = np.mean(1 - np.real(cdev), axis=axis)
    return acc

def bias(cest, ctrue=None, axis=0):
    cdev = _cdeviation(cest, ctrue=ctrue)
    bias = np.angle(np.mean(cdev, axis=axis))
    return bias
