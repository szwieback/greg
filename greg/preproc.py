'''
Created on Oct 25, 2021

@author: simon
'''

import numpy as np

import pyximport; pyximport.install()

def covariance_matrix(y):
    from cython_greg import covm
    y_shape = y.shape
    L = y_shape[-2]
    P = y_shape[-1]
    C_obs = covm(y.reshape((-1, L, P)))
    return C_obs.reshape((y_shape[:-2]) + (P, P))

def correlation(G_raw, inplace=False):
    # G_raw: Hermitian, [..., P, P]
    if inplace:
        G_out = G_raw
    else:
        G_out = np.copy(G_raw)
    P = G_raw.shape[-1]
    if G_raw.shape[-2] != P: raise ValueError('G needs to be square')
    ind_P = np.arange(P)
    G_diag = G_raw[..., ind_P, ind_P].real
    np.power(G_diag, -0.5, out=G_diag)
    G_out *= G_diag[..., :, np.newaxis]
    G_out *= G_diag[..., np.newaxis, :]
    return G_out

def force_doubly_nonnegative(G_raw, min_eig=0.0, inplace=False):
    # G_raw needs to be nonnegative P, P to begin with
    # adds a multiple of the identity so that the minimum eigenvalue is min_eig
    from cython_greg import fdd
    G_shape = G_raw.shape
    P = G_shape[-1]
    if G_shape[-2] != P: raise ValueError('G needs to be square')
    G_raw = G_raw.reshape((-1, P, P))  # should be a view
    if inplace:
        G_out = G_raw
    else:
        G_out = np.empty_like(G_raw)
    fdd(G_raw, G_out, min_eig=min_eig)
    return G_out.reshape(G_shape)

def force_doubly_nonnegative_py(G_raw, min_eig=0.0):
    # G_raw needs to be nonnegative P, P to begin with
    # adds a multiple of the identity so that the minimum eigenvalue is min_eig
    from scipy.linalg import eigh
    G_shape = G_raw.shape
    P = G_shape[-1]
    if G_shape[-2] != P: raise ValueError('G needs to be square')
    G_raw = G_raw.reshape((-1, P, P))  # should be a view
    N = G_raw.shape[0]
    G_out = G_raw.copy()
    for n in range(N):
        lam = eigh(
            G_raw[n, :, :], subset_by_index=[0, 0], eigvals_only=True)
        negshift = lam[0] - min_eig
        if negshift < 0:
            G_out[n, ...] -= negshift * np.eye(P) 
    return G_out.reshape(G_shape)

def valid_G(C_obs, corr=True):
    G = force_doubly_nonnegative(np.abs(C_obs))
    if corr:
        G = correlation(G, inplace=True)
    return G

