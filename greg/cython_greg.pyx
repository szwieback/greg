#!python
# cython: language_level=3

import numpy as np
# from scipy.linalg import eigh
from scipy.linalg import eigh
from pickle import FALSE

cimport cython

cdef extern from "complex.h":
    double complex conj(double complex)
    double cimag(double complex)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def covm(double complex[:, :, :] y):
    # G_raw needs to be nonnegative of size N, P, P and single precision
    # adds a multiple of the identity so that the minimum eigenvalue is min_eig

    cdef Py_ssize_t R = y.shape[0]
    cdef Py_ssize_t L = y.shape[1]
    cdef Py_ssize_t P = y.shape[2]

    cdef Py_ssize_t n, p1, p2, l

    C = np.zeros((R, P, P), dtype=np.complex128)
    cdef double complex [:, :, :] C_view = C
    Cr = np.zeros((P, P), dtype=np.complex128)
    cdef double complex [:, :] Cr_view = Cr


    for r in range(R):
        Cr_view[:, :] = 0.0
        for l in range(L):
            for p1 in range(P):
                for p2 in range(p1, P):
                    Cr_view[p1, p2] = Cr_view[p1, p2] + y[r, l, p1] * conj(y[r, l, p2])
        for p1 in range(P):
            for p2 in range(P):
                if p2 >= p1:
                    C_view[r, p1, p2] = Cr_view[p1, p2] / L
                else:
                    C_view[r, p1, p2] = conj(Cr_view[p2, p1]) / L
    return C

@cython.boundscheck(False)
@cython.wraparound(False)
def fdd(double[:, :, :] G_raw, double[:, :, :] G_out, double min_eig=0.0):
    # G_raw needs to be nonnegative of size N, P, P and single precision
    # adds a multiple of the identity so that the minimum eigenvalue is min_eig

    assert G_raw.shape[1] == G_raw.shape[2]
    cdef Py_ssize_t N = G_raw.shape[0]
    cdef Py_ssize_t P = G_raw.shape[1]

    cdef Py_ssize_t n, p

    lam = np.ones(1, dtype=np.double)
    cdef double [:] lam_view = lam

    for n in range(N):
        # may need to call lapack directly
        lam_view[:] = eigh(
            G_raw[n, :, :], subset_by_index=[0, 0], eigvals_only=True)[0]

        G_out[n, :, :] = G_raw[n, :, :]
        negshift = lam_view[0] - min_eig
        if negshift < 0:
            for p in range(P):
                G_out[n, p, p] -= negshift
    return G_out

@cython.boundscheck(False)
@cython.wraparound(False)
def _EMI(double complex[:, :, :] C_obs, double[:, :, :] G):
    # G_raw needs to be nonnegative of size N, P, P and single precision
    # adds a multiple of the identity so that the minimum eigenvalue is min_eig
    assert G.shape[1] == G.shape[2]
    cdef Py_ssize_t N = G.shape[0]
    cdef Py_ssize_t P = G.shape[1]
 
    cdef Py_ssize_t n, p1, p2
 
    ceig = np.empty((N, P), dtype=np.complex128)
    cdef double complex [:, :] ceig_view = ceig
    ceig_n = np.empty((P, 1), dtype=np.complex128)
    cdef double complex [:, :] ceig_n_view = ceig_n
    M1 = np.empty((P, P), dtype=np.complex128)
    M2 = np.empty((P, P), dtype=np.complex128)
    cdef double complex [:, :] M2_view = M2
    lam = np.ones(1, dtype=np.double)
    cdef double [:] lam_view = lam
    
    for n in range(N):
        # need to call lapack directly
        M1[:, :] = np.linalg.pinv(G[n, :, :])
        M2[:, :] = C_obs[n, :, :]            
        M1 *= M2
        # need to call lapack directly
        lam, ceig_n_view = eigh(M1, subset_by_index=[0, 0], eigvals_only=False)
        ceig_view[n, :] = ceig_n_view[:, 0]
    return ceig_view

@cython.boundscheck(False)
@cython.wraparound(False)
def _EVD(double complex[:, :, :] C_obs): 
    assert C_obs.shape[1] == C_obs.shape[2]
    cdef Py_ssize_t N = C_obs.shape[0]
    cdef Py_ssize_t P = C_obs.shape[1]
 
    cdef Py_ssize_t n, p1, p2
 
    ceig = np.empty((N, P), dtype=np.complex128)
    cdef double complex [:, :] ceig_view = ceig
    ceig_n = np.empty((P, 1), dtype=np.complex128)
    cdef double complex [:, :] ceig_n_view = ceig_n
    lam = np.ones(1, dtype=np.double)
    cdef double [:] lam_view = lam
    
    for n in range(N):
        # need to call lapack directly
        lam, ceig_n_view = eigh(C_obs[n, :, :], subset_by_index=[P-1, P-1], eigvals_only=False)
        ceig_view[n, :] = ceig_n_view[:, 0]
    return ceig_view


def _indtups(int P):
    cdef list indtups = []
    for indm in range(P):
        indtups.append((indm, indm))
    for indm in range(P):
        for inds in range(indm + 1, P):
            indtups.append((indm, inds))
    return indtups

@cython.boundscheck(False)
@cython.wraparound(False)
def _FI_magnitude(
        double [:, :, :] G, double complex[:, :, :] C_obs, double complex[:, :] cphases, long L):
    # cphases is assumed to be normalized
    cdef Py_ssize_t P = G.shape[2]
    cdef list indtups = _indtups(P)
    cdef Py_ssize_t N = len(indtups)
    cdef Py_ssize_t M = G.shape[0]
    FGG = np.zeros((M, N, N), dtype=np.float64)
    FGt = np.zeros((M, N, P), dtype=np.float64)
    cdef double [:, :, :] FGG_view = FGG
    cdef double [:, :, :] FGt_view = FGt
    cdef double val = 0.0
    cdef (long, long) indtup1 = (0, 0)
    cdef (long, long) indtup2 = (0, 0)
    for jindtup1 in range(N):
        indtup1 = indtups[jindtup1]
        for jindtup2 in range(jindtup1, N):
            indtup2 = indtups[jindtup2]
            for m in range(M):
                if indtup1[0] != indtup1[1] and indtup2[0] != indtup2[1]:
                    val = -2 * L * (
                        G[m, indtup1[0], indtup2[1]] * G[m, indtup2[0], indtup1[1]]
                        +G[m, indtup1[0], indtup2[0]] * G[m, indtup2[1], indtup1[1]])
                elif indtup1[0] == indtup1[1] and indtup2[0] == indtup2[1]:
                    val = -L * (G[m, indtup1[0], indtup2[0]] * G[m, indtup1[0], indtup2[0]])
                elif indtup1[0] != indtup1[1] and indtup2[0] == indtup2[1]:
                    val = -2 * L * (G[m, indtup1[0], indtup2[0]] * G[m, indtup2[0], indtup1[1]])
                elif indtup1[0] == indtup1[1] and indtup2[0] != indtup2[1]:
                    val = -2 * L * (G[m, indtup2[0], indtup1[0]] * G[m, indtup1[0], indtup2[1]])
                else:
                    val = np.nan
                FGG_view[m, jindtup1, jindtup2] = val
                FGG_view[m, jindtup2, jindtup1] = val
        if indtup1[0] != indtup1[1]:
            for m in range(M):
                val = - 2 * L * cimag(C_obs[m, indtup1[0], indtup1[1]] * conj(cphases[m, indtup1[0]])
                          * cphases[m, indtup1[1]])     
                for p in range(P):
                    FGt_view[m, jindtup1, p] = val * ((indtup1[0] == p) - (indtup1[1] == p))
    return FGG_view, FGt_view


@cython.boundscheck(False)
@cython.wraparound(False)
def _FI_magnitude_offdiagonal(
        double [:, :, :] G, double complex[:, :, :] C_obs, double complex[:, :] cphases, long L):
    # cphases is assumed to be normalized
    cdef Py_ssize_t P = G.shape[2]
    cdef list indtups = _indtups(P)[P:] # offdiagonal only
    cdef Py_ssize_t N = len(indtups)
    cdef Py_ssize_t M = G.shape[0]
    FGG = np.zeros((M, N, N), dtype=np.float64)
    FGt = np.zeros((M, N, P), dtype=np.float64)
    cdef double [:, :, :] FGG_view = FGG
    cdef double [:, :, :] FGt_view = FGt
    cdef double val = 0.0
    cdef (long, long) indtup1 = (0, 0)
    cdef (long, long) indtup2 = (0, 0)
    for jindtup1 in range(N):
        indtup1 = indtups[jindtup1]
        for jindtup2 in range(jindtup1, N):
            indtup2 = indtups[jindtup2]
            for m in range(M):
                if indtup1[0] != indtup1[1] and indtup2[0] != indtup2[1]:
                    val = -2 * L * (
                        G[m, indtup1[0], indtup2[1]] * G[m, indtup2[0], indtup1[1]]
                        +G[m, indtup1[0], indtup2[0]] * G[m, indtup2[1], indtup1[1]])
                FGG_view[m, jindtup1, jindtup2] = val
                FGG_view[m, jindtup2, jindtup1] = val
        for m in range(M):
            val = - 2 * L * cimag(C_obs[m, indtup1[0], indtup1[1]] * conj(cphases[m, indtup1[0]])
                      * cphases[m, indtup1[1]])     
            for p in range(P):
                FGt_view[m, jindtup1, p] = val * ((indtup1[0] == p) - (indtup1[1] == p))
    return FGG_view, FGt_view