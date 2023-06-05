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
    if G is None:
        C_shape
        G = force_doubly_nonnegative_py(np.abs(C_obs).real)
    G = G.reshape((-1, P, P))
    N = G.shape[0]
    if corr:
        G = correlation(G, inplace=False)
        C_obs = correlation(C_obs, inplace=False)
    ceig = np.empty((N, P), dtype=np.complex128)
    for n in range(N):
        _, ceig_n = eigh(C_obs[n, :, :], subset_by_index=[
                         P-1, P-1], eigvals_only=False)
        ceig[n, :] = ceig_n[:, 0] * \
            (ceig_n[0, 0].conj() / np.abs(ceig_n[0, 0]))
    return ceig.reshape(C_shape[:-1])


if __name__ == '__main__':
    C_obs = np.array([[3, 1j], [-1j, 4]])
    ceig = EMI_py(C_obs)
