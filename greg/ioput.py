'''
Created on Dec 1, 2021

@author: szwieback
'''

import os
import pickle
import zlib
import numpy as np

def enforce_directory(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            pass

def save_object(obj, filename):
    enforce_directory(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        f.write(
            zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)))

def load_object(filename):
    if os.path.splitext(filename)[1].strip() == '.npy':
        return np.load(filename)
    with open(filename, 'rb') as f:
        obj = pickle.loads(zlib.decompress(f.read()))
    return obj
