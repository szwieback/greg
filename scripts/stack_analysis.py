'''
Created on Mar 28, 2022

@author: simon
'''
import os
import numpy as np
from itertools import combinations


def read_cphases(path0, rtype):
    phases = np.load(os.path.join(path0, 'proc', rtype, 'phases.npy'))
    cphases = np.exp(1j*phases)
    return cphases

    

def compare_cphases(cp, cpr, comptype='all'):
    from greg import circular_accuracy
    axes = {'all': None, 'time': (0, 1), 'space': 2}
    return circular_accuracy(cp, cpr, axis=axes[comptype])

    

if __name__ == '__main__':
    
    path0 = '/home/simon/Work/greg/stacks/'
#     rtypes = ['none', 'hadamard', 'spectral', 'hadspec']
#     cphases = {rtype: read_cphases(path0, rtype) for rtype in rtypes}
#     for rtype1, rtype2 in combinations(rtypes, 2):
#         acc = compare_cphases(cphases[rtype1], cphases[rtype2], comptype='all')
#         print(rtype1, rtype2, acc) # convert to equivalent phase sd

    # implement a separate uncertainty estimation based on expected Kp
    # merge into greglater
    rtype = 'hadamard'
    cphases = read_cphases(path0, rtype)
    import matplotlib.pyplot as plt
    plt.imshow(np.angle(cphases[..., -10]))
    plt.show()


