'''
Created on Mar 29, 2022

@author: simon
'''
import numpy as np
import warnings

def _A(P, clip_first=True):
    if clip_first:
        A = np.concatenate((-np.ones((P - 1, 1)), np.eye(P-1)), axis=1)
#         A = np.zeros((P - 1, P))
#         A[:, 0] = -1
#         A[:, 1:] = np.eye(P - 1)
    else:
        raise NotImplementedError(f"Variable clip_first set to False")
    return A

def _C_phase_differences(C_all):
    A = _A(C_all.shape[-1])
    C = A @ C_all @ A.T # implicit broadcasting
    return C

def C_expected_partial(G, L):
    I = np.eye(G.shape[-1])[np.newaxis, ...]
    F = -2 * L * (np.linalg.inv(G) * G - I)
    C_all = np.linalg.pinv(-F, hermitian=True)
    C = _C_phase_differences(C_all)
    return C
    
def phases_covariance(G=None, C_obs=None, L=None, method='expected_partial'):
    if method != 'expected_partial':
        raise NotImplementedError(
            f"Phase history covariance method {method} not implemented")
    if L is None:
        L = 50
        warnings.warn(f"L not provided, assuming L = {L}")
    shape = G.shape  # may need to take C_obs in the future
    if G is not None and C_obs is not None:
        assert G.shape == C_obs.shape
    P = shape[-1]
    if shape[-2] != P: raise ValueError('G and C_obs need to be square')
    G = G.reshape((-1, P, P))
    assert method == 'expected_partial'
    C = C_expected_partial(G, L)
    C = C.reshape(shape[:-2] + C.shape[-2:])
    return C

