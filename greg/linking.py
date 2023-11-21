'''
Created on Oct 25, 2021

@author: simon
'''
import numpy as np

from greg.preproc import (
    correlation, force_doubly_nonnegative, force_doubly_nonnegative_py)


def EMI(C_obs, G=None, corr=True):
    from greg.cython_greg import _EMI
    # output magnitude not normalized
    C_shape = C_obs.shape
    if G is not None:
        assert G.shape == C_shape
    P = C_shape[-1]
    if C_shape[-2] != P:
        raise ValueError('G needs to be square')
    C_obs = C_obs.reshape((-1, P, P))
    if G is None:
        G = force_doubly_nonnegative(np.abs(C_obs).real, inplace=True)
    G = G.reshape((-1, P, P))
    if corr:
        G = correlation(G, inplace=False)
        C_obs = correlation(C_obs, inplace=False)
    ceig = np.array(_EMI(C_obs, G))
    ceig *= (ceig[:, 0].conj() / np.abs(ceig[:, 0]))[:, np.newaxis]
    return ceig.reshape(C_shape[:-1])


def EMI_py(C_obs, G=None, corr=True):
    from scipy.linalg import eigh
    C_shape = C_obs.shape
    if G is not None:
        assert G.shape == C_shape
    P = C_shape[-1]
    if C_shape[-2] != P:
        raise ValueError('G needs to be square')
    C_obs = C_obs.reshape((-1, P, P))
    if G is None:
        G = force_doubly_nonnegative_py(np.abs(C_obs).real)
    G = G.reshape((-1, P, P))
    N = G.shape[0]
    if corr:
        G = correlation(G, inplace=False)
        C_obs = correlation(C_obs, inplace=False)
    ceig = np.empty((N, P), dtype=np.complex128)
    for n in range(N):
        had = C_obs[n, :, :]
        had *= np.linalg.pinv(G[n, :, :])
        _, ceig_n = eigh(
            had, subset_by_index=[0, 0], eigvals_only=False)
        ceig[n, :] = ceig_n[:, 0] * \
            (ceig_n[0, 0].conj() / np.abs(ceig_n[0, 0]))
    return ceig.reshape(C_shape[:-1])


def EVD_py(C_obs, G=None, corr=True):
    from scipy.linalg import eigh
    C_shape = C_obs.shape
    if G is not None:
        assert G.shape == C_shape
    P = C_shape[-1]
    if C_shape[-2] != P:
        raise ValueError('G needs to be square')
    C_obs = C_obs.reshape((-1, P, P))
    N = C_obs.shape[0]
    if corr:
        C_obs = correlation(C_obs, inplace=False)
    ceig = np.empty((N, P), dtype=np.complex128)
    if G is None:
        _C_obs = C_obs
    else:
        G = G.reshape((-1, P, P))
        if corr:
            G = correlation(G, inplace=False)
        _C_obs = C_obs / np.abs(C_obs) * G # do not mess with negative eigenvalues
    for n in range(N): # needs cython port
        _, ceig_n = eigh(_C_obs[n, :, :], subset_by_index=[
                         P-1, P-1], eigvals_only=False)
        ceig[n, :] = ceig_n[:, 0] * (
            ceig_n[0, 0].conj() / np.abs(ceig_n[0, 0]))
    return ceig.reshape(C_shape[:-1])

def EVD(C_obs, G=None, corr=True):
    from greg.cython_greg import _EVD
    # output magnitude not normalized
    C_shape = C_obs.shape
    if G is not None: assert G.shape == C_shape
    P = C_shape[-1]
    if C_shape[-2] != P: raise ValueError('G needs to be square')
    C_obs = C_obs.reshape((-1, P, P))
    if G is None:
        if corr:
            _C_obs = correlation(C_obs, inplace=False)
        else:
            _C_obs = C_obs
    else:
        G = G.reshape((-1, P, P))
        if corr:
            G = correlation(G, inplace=False)    
        _C_obs = C_obs / np.abs(C_obs) * G
    invalid = np.any(np.logical_not(np.isfinite(_C_obs)), axis=(1, 2))
    _C_obs[invalid, ...] = 1
    ceig = np.array(_EVD(_C_obs))
    ceig *= np.conj(ceig[:, 0] / np.abs(ceig[:, 0]))[:, np.newaxis]
    ceig[invalid, ...] = np.nan  
    return ceig.reshape(C_shape[:-1])


if __name__ == '__main__':
    eps = 0.1
    C_obs = np.array([[3, 1j, eps + 1j], [-1j, 4, 1], [eps -1j, 1, 4]])
    ceig = EVD(C_obs)
    ceig_py = EVD_py(C_obs)
    print(ceig * ceig_py.conj())
