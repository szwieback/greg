'''
Created on Nov 12, 2021

@author: simon
'''

import numpy as np
from collections import namedtuple
np.set_printoptions(precision=2, suppress=True)

from linking import EMI_py


def ssqrtm(A, inverse=False):
    W, V = np.linalg.eigh(A)
    if inverse:
        np.power(W, -0.5, out=W)
    else:
        np.power(W, 0.5, out=W)
    np.matmul(V * W[..., np.newaxis,:], np.swapaxes(V, -1, -2).conj(), out=V)
    return V

# def cca_W_old(C, M1, d = 1):
#     C12w = (ssqrtm(C[0:M1, 0:M1], inverse=True) @ C[0:M1, M1:M]
#              @ ssqrtm(C[M1:M, M1:M], inverse=True))
#     V1, rho, V2T = np.linalg.svd(C12w)
#     U1 = ssqrtm(C[0:M1, 0:M1], inverse=True) @ V1
#     U2 = ssqrtm(C[M1:, M1:], inverse=True) @ V2T.T
#     
#     U1d = U1[:, :d]
#     rhod = rho[:d]
#     U2d = U2[:, :d]
#     
#     W1 = C[0:M1, 0:M1] @ U1d @ np.diag(np.sqrt(rhod))
#     W2 = C[M1:, M1:] @ U2d @ np.diag(np.sqrt(rhod))
#     Psi1 = C[0:M1, 0:M1] - W1 @ W1.T
#     Psi2 = C[M1:, M1:] - W2 @ W2.T
#     return W1, W2


def cca_W_(C, M1, d=1):
    if len(C.shape) > 3: raise NotImplementedError
    M = C.shape[-1]
    if M1 >= M: raise ValueError(f'M1 {M1} exceeds dimension of C {M}')
    if M != C.shape[-2]: raise ValueError(f'C is not square')

    Cisqr1 = ssqrtm(C[...,:M1,:M1], inverse=True)
    Cisqr2 = ssqrtm(C[..., M1:, M1:], inverse=True)

    C12w = np.matmul(Cisqr1, C[...,:M1, M1:])
    np.matmul(C12w, Cisqr2, out=C12w)
    V1, rho, V2h = np.linalg.svd(C12w, hermitian=False)
    U1 = np.matmul(Cisqr1, V1[...,:d])
    U2 = np.matmul(Cisqr2, np.swapaxes(V2h, -1, -2)[...,:d].conj())
    del Cisqr1, Cisqr2
    sqrt_rho = np.power(rho[...,:d], 0.5)
    del V1, V2h, rho
    W1 = np.matmul(C[...,:M1,:M1], U1 * sqrt_rho[..., np.newaxis,:])
    W2 = np.matmul(C[..., M1:, M1:], U2 * sqrt_rho[..., np.newaxis,:])
    return W1, W2


def cca_2lv(C, M1):
    # with 2 latent variables
    d = 1  # must be
    if len(C.shape) > 3: raise NotImplementedError
    M = C.shape[-1]
    if M1 >= M: raise ValueError(f'M1 {M1} exceeds dimension of C {M}')
    if M != C.shape[-2]: raise ValueError(f'C is not square')

    Cisqr1 = ssqrtm(C[...,:M1,:M1], inverse=True)  # redundant for step >= 1
    Cisqr2 = ssqrtm(C[..., M1:, M1:], inverse=True)

    C12w = np.matmul(Cisqr1, C[...,:M1, M1:])
    np.matmul(C12w, Cisqr2, out=C12w)
    V1, rho, V2h = np.linalg.svd(C12w)
    V1 = V1[...,:d]
    V2 = np.swapaxes(V2h, -1, -2)[...,:d].conj()
    U1 = np.matmul(Cisqr1, V1)
    U2 = np.matmul(Cisqr2, V2)
    del Cisqr2
    rho = rho[...,:d]
    W1 = np.matmul(C[...,:M1,:M1], U1)
    # W2_tilde = np.matmul(C[..., M1:, M1:], U2 * rho[..., np.newaxis, :])
    W2 = np.matmul(C[..., M1:, M1:], U2)
    B = rho[..., np.newaxis]
    Phi2 = (1 - np.power(rho, 2))[..., np.newaxis]  # conditional var | z1    
    return W1, W2, B, Phi2, U1, Cisqr1


def link_step():
    Czx = np.zeros(
        C_.shape[:-2] + (W2.shape[-2] + 1,) * 2, dtype=C_.dtype)
    Czx[..., 0, 0] = 1
    Czx[..., 1:, 1:] = C_[..., M1:, M1:]
    Czx[..., 1:, 0] = B[..., 0,:] * W2[..., 0]
    Czx[..., 0, 1:] = Czx[..., 1:, 0].conj()
    # compute ceig
    ceig2 = EMI_py(Czx)
    # adjust phase offset
    # cref = np.sum(ceig_list[-1]* W1[...,0], axis=-1)
    # cref = np.sum(ceig_list[-1].conj() * W1[..., 0], axis=-1)
    cref = ceig_list[-1][..., -1] * W1[..., -1, 0].conj()
    cref /= np.abs(cref)
    ceig2 *= cref[..., np.newaxis]

    
def test():

    M = 8
    M1 = M // 2
    d = 1
    from simulation import decay_model
    y = decay_model(P=M, R=50, coh_decay=0.7, coh_infty=0.4)
    C = np.mean(y[..., np.newaxis] * y.conj()[..., np.newaxis,:], axis=1)
    # W1, W2 = cca_W_(C, M1, d=d)
    # Sigma_est = C.copy()
    # Sigma_est[..., :M1, M1:] = np.matmul(W1, np.swapaxes(W2, -2, -1).conj())
    # Sigma_est[..., M1:, :M1] = np.swapaxes(Sigma_est[..., :M1, M1:], -2, -1).conj()
    W1, W2, B, _, U1, U2 = cca_2lv(C, M1)
    Sigma_est = C.copy()

    Sigma_est[...,:M1, M1:] = np.matmul(W1, np.swapaxes(np.matmul(W2, B), -2, -1).conj())
    Sigma_est[..., M1:,:M1] = np.swapaxes(Sigma_est[...,:M1, M1:], -2, -1).conj()
    print(U1.shape, U2.shape)
    rho_2 = np.matmul(np.swapaxes(U1, -2, -1).conj(), np.matmul(C[...,:M1, M1:], U2)).real
    
    print(B[0:10].flatten())
    print(rho_2[0:10].flatten())


def test_sequential():

    M = 32
    M1 = 4
    d = 1
    steps = M // M1 - 1
    
    from simulation import decay_model
    y = decay_model(P=M, R=50, coh_decay=0.7, coh_infty=0.4)
    C = np.mean(y[..., np.newaxis] * y.conj()[..., np.newaxis,:], axis=1)

    W_list = []
    B_list = []
    ceig_list = []
    for step in range(steps):
        M_start = step * M1
        M_end = (step + 2) * M1 if (step + 2) * M1 <= M else M
        C_ = C[..., M_start:M_end, M_start:M_end]
        W1, W2, B, _, U1, Cisqr1 = cca_2lv(C_, M1)
        W_list.append(W1)
        if step == 0:
            U1_C_cross_old, B_old, C_old = None, None, None
            ceig1 = EMI_py(C_[...,:M1,:M1])
            ceig2 = np.zeros(1)  # fix this later
            ceig1 = EMI_py(C_)
            ceig_list.append(ceig1[...,:M1])
            ceig_list.append(ceig1[..., M1:])
        elif step > 1:
            B_revised = np.matmul(U1_C_cross_old, U1).real  # can be <0
            B_list.append(B_revised)
            link_step()
            # return theta
            ceig_list.append(ceig2[..., 1:])
        if step == steps - 1:
            B_list.append(B)
            W_list.append(W2)
            # add last theta
        U1_C_cross_old = np.matmul(
            np.swapaxes(U1, -2, -1).conj(), C_[...,:M1, M1:])
        # B_old = B
        # C_old = C_
        # print(W1_old.shape)
    ceig = np.concatenate(ceig_list, axis=-1)
    print(np.angle(ceig[1,:]))
    print(np.angle(EMI_py(C)[1,:]))
    # major phase linking issues to do with inplace
    # check signs and referencing
    
    # Sigma_est = C.copy()
    # Sigma_est[...,:M1, M1:] = np.matmul(W1, np.swapaxes(np.matmul(W2, B), -2, -1).conj())
    # Sigma_est[..., M1:,:M1] = np.swapaxes(Sigma_est[...,:M1, M1:], -2, -1).conj()
    # rho_2 = np.matmul(np.swapaxes(U1, -2, -1).conj(), np.matmul(C[..., :M1, M1:], U2)).real
    #
    #
    # print(B[0:10].flatten())
    # print(rho_2[0:10].flatten())


if __name__ == '__main__': 
    test_sequential()
