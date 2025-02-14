'''
Created on Mar 29, 2022

@author: simon
'''
import numpy as np
import warnings

FI_methods = ('expected_partial', 'observed_partial', 'full', 'full_python', 
              'full_offdiagonal', 'full_offdiagonal_python')

def _A(P, clip_first=True):
    if clip_first:
        A = np.concatenate((-np.ones((P - 1, 1)), np.eye(P - 1)), axis=1)
    else:
        raise NotImplementedError(f"Variable clip_first set to False")
    return A

def _C_phase_differences(C_all):
    A = _A(C_all.shape[-1])
    C = A @ C_all @ A.T  # implicit broadcasting
    return C

def FI_expected_partial(G, L=30):
    I = np.eye(G.shape[-1])[np.newaxis, ...]
    F = -2 * L * (np.linalg.pinv(G) * G - I)
    return F

def C_expected_partial(G, L):
    F = FI_expected_partial(G, L)
    C_all = np.linalg.pinv(-F, hermitian=True)
    C = _C_phase_differences(C_all)
    return C

def phases_covariance(G, C_obs=None, L=None, cphases=None, method='expected_partial'):
    if L is None:
        L = 50
        warnings.warn(f"L not provided, assuming L = {L}")
    shape = G.shape
    if G is not None and C_obs is not None:
        assert G.shape == C_obs.shape
    P = shape[-1]
    if shape[-2] != P: raise ValueError('G and C_obs need to be square')
    _G = G.reshape((-1, P, P))
    _cphases = cphases.reshape((-1, P)) if cphases is not None else None
    _C_obs = C_obs.reshape((-1, P, P)) if C_obs is not None else None
    C = C_phase_history(_G, cphases, L=L, C_obs=C_obs, normalize=True, method=method)
    C = C.reshape(shape[:-2] + C.shape[-2:])
    return C

def FI_theta_theta(G, C_obs, cphases, L=30):
    # cphases is assumed to be normalized
    cphases[np.isnan(cphases)] = 1
    had = np.linalg.pinv(G) * C_obs
    B = (cphases[..., np.newaxis].conj() * had * cphases[..., np.newaxis,:])
    diagterm = L * np.real(np.sum(B, axis=1) + np.sum(B, axis=2))
    F = -L * np.real(B + np.swapaxes(B, 1, 2))
    indices = np.arange(B.shape[1])
    F[:, indices, indices] += diagterm
    return F

def FI_magnitude(G, C_obs, cphases, L=30):
    # cphases is assumed to be normalized
    P = G.shape[-1]
    indtups = _indtups(P)
    N = len(indtups)
    FGG = np.zeros((G.shape[0], N, N))
    FGt = np.zeros((G.shape[0], N, P))
    for jindtup1, indtup1 in enumerate(indtups):
        for jindtup2 in range(jindtup1, N):
            indtup2 = indtups[jindtup2]
            if indtup1[0] != indtup1[1] and indtup2[0] != indtup2[1]:
                val = -2 * L * (
                    G[:, indtup1[0], indtup2[1]] * G[:, indtup2[0], indtup1[1]]
                    +G[:, indtup1[0], indtup2[0]] * G[:, indtup2[1], indtup1[1]])
            elif indtup1[0] == indtup1[1] and indtup2[0] == indtup2[1]:
                val = -L * (G[:, indtup1[0], indtup2[0]] * G[:, indtup1[0], indtup2[0]])
            elif indtup1[0] != indtup1[1] and indtup2[0] == indtup2[1]:
                val = -2 * L * (G[:, indtup1[0], indtup2[0]] * G[:, indtup2[0], indtup1[1]])
            elif indtup1[0] == indtup1[1] and indtup2[0] != indtup2[1]:
                val = -2 * L * (G[:, indtup2[0], indtup1[0]] * G[:, indtup1[0], indtup2[1]])
            else:
                val = np.nan
            FGG[:, jindtup1, jindtup2] = val
            FGG[:, jindtup2, jindtup1] = val
        if indtup1[0] != indtup1[1]:
            fac = - 2 * L * np.imag(C_obs[:, indtup1[0], indtup1[1]] * cphases[:, indtup1[0]].conj()
                      * cphases[:, indtup1[1]])            
            for p in range(P):
                FGt[:, jindtup1, p] = fac * ((indtup1[0] == p) - (indtup1[1] == p))
    return FGG, FGt

def FI_magnitude_offdiagonal(G, C_obs, cphases, L=30):
    # only offdiagonal Ginv
    # cphases is assumed to be normalized
    P = G.shape[-1]
    indtups = _indtups(P)[P:]  # only offdiagonal
    N = len(indtups)
    FGG = np.zeros((G.shape[0], N, N))
    FGt = np.zeros((G.shape[0], N, P))
    for jindtup1, indtup1 in enumerate(indtups):
        for jindtup2 in range(jindtup1, N):
            indtup2 = indtups[jindtup2]
            if indtup1[0] != indtup1[1] and indtup2[0] != indtup2[1]:
                val = -2 * L * (
                    G[:, indtup1[0], indtup2[1]] * G[:, indtup2[0], indtup1[1]]
                    +G[:, indtup1[0], indtup2[0]] * G[:, indtup2[1], indtup1[1]])
            FGG[:, jindtup1, jindtup2] = val
            FGG[:, jindtup2, jindtup1] = val
        if indtup1[0] != indtup1[1]:
            fac = - 2 * L * np.imag(C_obs[:, indtup1[0], indtup1[1]] * cphases[:, indtup1[0]].conj()
                      * cphases[:, indtup1[1]])
            for p in range(P):
                FGt[:, jindtup1, p] = fac * ((indtup1[0] == p) - (indtup1[1] == p))
    return FGG, FGt

def FI_magnitude_benchmark(G, C_obs, cphases, L=30):
    # old implementation; treats diagonal and off-diagonal (mag) separately
    # cphases is assumed to be normalized
    # tested with numerical evaluation of derivative
    P = G.shape[-1]
    indtups = _indtups(P)
    indtups_mag = indtups[P:]
    FGG = np.zeros((G.shape[0], len(indtups), len(indtups)))
    FGt = np.zeros((G.shape[0], len(indtups), P))

    N = len(indtups_mag)

    FI_mag = np.zeros((G.shape[0], N, N))
    FI_diag = np.zeros((G.shape[0], P, P))
    FI_linked_diag = np.zeros((G.shape[0], P, P))
    FI_linked_mag = np.zeros((G.shape[0], P, N))
    FI_mag_diag = np.zeros((G.shape[0], N, P))
    for jindtup1, indtup1 in enumerate(indtups_mag):
        for jindtup2, indtup2 in enumerate(indtups_mag):
            FI_mag[:, jindtup1, jindtup2] = -2 * L * (
                G[:, indtup1[0], indtup2[1]] * G[:, indtup2[0], indtup1[1]]
                +G[:, indtup1[0], indtup2[0]] * G[:, indtup2[1], indtup1[1]])
        fac = -2 * L * np.imag(C_obs[:, indtup1[0], indtup1[1]] * cphases[:, indtup1[0]].conj()
                              * cphases[:, indtup1[1]])
        for jdiag in range(P):
            FI_mag_diag[:, jindtup1, jdiag] = -2 * L * (
                                            G[:, indtup1[0], jdiag] * G[:, jdiag, indtup1[1]])
            FI_linked_mag[:, jdiag, jindtup1] = fac * (
                (indtup1[0] == jdiag) - (indtup1[1] == jdiag))
    for jdiag1 in range(P):
        for jdiag2 in range(P):
            FI_diag[:, jdiag1, jdiag2] = -L * G[:, jdiag1, jdiag2] * G[:, jdiag2, jdiag1]
    FGG[:,:P,:P] = FI_diag
    FGG[:, P:P + N, P:P + N] = FI_mag
    FGG[:, P:P + N,:P] = FI_mag_diag
    FGG[:,:P, P:P + N] = np.swapaxes(FI_mag_diag, 1, 2)
    FGt[:, P:P + N,:] = np.swapaxes(FI_linked_mag, 1, 2)
    return FGG, FGt

def C_phase_history(G, cphases, L=30, C_obs=None, normalize=True, method='full'):
    # G and C should be [M, P, P]
    if C_obs is None:
        C_obs = G
        warnings.warn("C_obs not provided")
    cphases_norm = cphases / np.abs(cphases) if normalize else cphases
    if 'full' in method:
        if method == 'full':
            from greg.cython_greg import _FI_magnitude
            FGG, FGt = _FI_magnitude(G, C_obs, cphases_norm, L)
        elif method == 'full_python':
            FGG, FGt = FI_magnitude(G, C_obs, cphases_norm, L=L)
        elif method == 'full_offdiagonal_python':
            FGG, FGt = FI_magnitude_offdiagonal(G, C_obs, cphases_norm, L=L)
        elif method == 'full_offdiagonal':
            from greg.cython_greg import _FI_magnitude_offdiagonal
            FGG, FGt = _FI_magnitude_offdiagonal(G, C_obs, cphases_norm, L)
        elif method == 'full_benchmark':  # old implementation, thoroughly tested
            FGG, FGt = FI_magnitude_benchmark(G, C_obs, cphases_norm, L=L)
        else:
            raise ValueError(f"Unrecognized method {method}")
        Ftt = FI_theta_theta(G, C_obs, cphases_norm, L=L)
        K = covariance_schur_complement(Ftt, FGG, FGt)
    else:
        if method == 'expected_partial':
            F = FI_expected_partial(G, L=L)
            K = np.linalg.pinv(-F, hermitian=True) # guaranteed to be nonnegative definite
        elif method == 'observed_partial':
            F = FI_theta_theta(G, C_obs, cphases_norm, L=L)
            K = pinv_abs(-F)
        else:
            raise ValueError(f"Unrecognized method {method}")
    C = _C_phase_differences(K)
    return C

def _indtups(P):
    indtups = []
    for indm in range(P):
        indtups.append((indm, indm))
    for indm in range(P):
        for inds in range(indm + 1, P):
            indtups.append((indm, inds))
    return indtups

def pinv_abs(F, thresh = 1e-4):
    # pseudoinverse, is positive semi-definite (by taking abs of eigenvalues)
    # assumes F is hermitian
    w, v = np.linalg.eigh(F)
    wi = np.zeros_like(w)
    ind = np.abs(w) > thresh 
    wi[ind] = np.abs(w[ind]) ** (-1)
    K = np.einsum('...ij,...kj', v, wi[..., np.newaxis, :] * v, optimize='greedy')    
    return K

def covariance_schur_complement(Ftt, FGG, FGt):
    FGGi = np.linalg.pinv(FGG, hermitian=True)
    Fb = np.einsum('...ji,...jk,...kl->...il', FGt, FGGi, FGt, optimize='greedy')
    Fsc = Ftt - Fb # should be negative semidefinite, but not guaranteed for arbitrary estimates
    # K = np.linalg.pinv(-Fsc, hermitian=True)
    K = pinv_abs(-Fsc)
    return K


if __name__ == '__main__':
    from greg import circular_normal, covariance_matrix, valid_G, regularize_G, EMI, decay_model, specreg, hadreg, correlation
    # Sigma = np.array([[1, 1j * 0.50, 1j * 0.25, 1j * 0.125], [-1j * 0.50, 1, 0.50, 0.25], [-1j * 0.25, 0.5, 1, 0.5], [-1j * 0.125, 0.25, 0.5, 1.0]])

    L = 32 #32
    rng = np.random.default_rng(1)

    # y = circular_normal([2048, L, 4], Sigma=Sigma, rng=rng)
    y = decay_model(R=256, L=L, P=10, coh_decay=0.6, rng=rng)
    y[..., 0:2] *= -1
    C_obs = correlation(covariance_matrix(y))
    G0 = valid_G(C_obs, corr=True)
    # G = hadreg(G0, alpha=0.50, nu=0.05)
    G = specreg(G0, beta=0.0)
    cphases = EMI(C_obs, G=G, corr=True)
    cphases_norm = cphases / np.abs(cphases)
    

    C_f = lambda method: C_phase_history(G, cphases, C_obs=C_obs, L=L, method=method)

    Cs = {}
    method_ref = 'full_benchmark'
    methods = ['expected_partial', 'observed_partial', 'full', 'full_python', 'full_offdiagonal', 'full_offdiagonal_python', 'full_benchmark']
    for method in methods:
        Cs[method] = C_f(method)
        
    # import timeit
    # for method in methods:
    #     C_fm = lambda: C_f(method)
    #     time_taken = timeit.timeit(C_fm, number=3)
    #     print(f"{method}, {time_taken:.2f}")
    #

    np.set_printoptions(precision=3, suppress=True)    
    for method in methods:
        dC = Cs[method] - Cs[method_ref]
        eigvd = np.linalg.eigh(dC)[0]
        eigv = np.linalg.eigh(Cs[method])[0]
        print(method, np.mean(eigvd, axis=0), np.mean(eigv, axis=0))
        

