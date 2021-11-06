'''
Created on Oct 25, 2021

@author: simon
'''
import numpy as np

from greg import correlation, force_doubly_nonnegative

from greg.cython_greg import _EMI


def EMI(C_obs, G=None, corr=True):
    # output magnitude not normalized
    C_shape = C_obs.shape
    if G is not None:
        assert G.shape == C_shape
    P = C_shape[-1]
    if C_shape[-2] != P: raise ValueError('G needs to be square')
    C_obs = C_obs.reshape((-1, P, P))
    if G is None:
        G = force_doubly_nonnegative(np.abs(C_obs).real, inplace=True)
    G = G.reshape((-1, P, P))    
    if corr:
        G = correlation(G, inplace=True)
        C_obs = correlation(C_obs, inplace=True)
    ceig = np.array(_EMI(C_obs, G))
    ceig *= (ceig[:, 0].conj())[:, np.newaxis]
    return ceig.reshape(C_shape[:-1])


if __name__ == '__main__':
    C_obs = np.array([[3,1j], [-1j, 4]])
    EMI(C_obs)
