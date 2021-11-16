'''
Created on Nov 12, 2021

@author: simon
'''
import numpy as np
np.set_printoptions(precision=3, suppress=True)

def ssqrtm(A, inverse=False):
    W, V = np.linalg.eigh(A)
    if inverse:
        np.power(W, -0.5, out=W)
    else:
        np.power(W, 0.5, out=W)
    np.matmul(V * W[..., np.newaxis, :], np.swapaxes(V, -1, -2).conj(), out=V)
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

    Cisqr1 = ssqrtm(C[..., :M1, :M1], inverse=True)
    Cisqr2 = ssqrtm(C[..., M1:, M1:], inverse=True)

    C12w = np.matmul(Cisqr1, C[..., :M1, M1:])
    np.matmul(C12w, Cisqr2, out=C12w)
    V1, rho, V2h = np.linalg.svd(C12w, hermitian=False)
    U1 = np.matmul(Cisqr1, V1[..., :d])
    U2 = np.matmul(Cisqr2, np.swapaxes(V2h, -1, -2)[..., :d].conj())
    del Cisqr1, Cisqr2
    sqrt_rho = np.power(rho[..., :d], 0.5)
    del V1, V2h, rho
    W1 = np.matmul(C[..., :M1, :M1], U1 * sqrt_rho[..., np.newaxis, :])
    W2 = np.matmul(C[..., M1:, M1:], U2 * sqrt_rho[..., np.newaxis, :])
    return W1, W2

def test():

    M = 10
    M1 = 5
    d = 1
    from simulation import decay_model
    from preproc import covariance_matrix
    C = covariance_matrix(decay_model(P=10, coh_decay=0.7, coh_infty=0.4))
    W1, W2 = cca_W_(C, M1, d=d)
    
    Sigma_est = C.copy()
    Sigma_est[..., :M1, M1:] = np.matmul(W1, np.swapaxes(W2, -2, -1).conj())
    Sigma_est[..., M1:, :M1] = np.swapaxes(Sigma_est[..., :M1, M1:], -2, -1).conj()
#     print(np.matmul(np.linalg.inv(Sigma_est), C)[0, ...])
    print(W1[0, :, 0])
    
if __name__ == '__main__': 
    test()