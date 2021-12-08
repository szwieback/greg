'''
Created on Oct 27, 2021

@author: simon
'''
from numpy.random import default_rng
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from collections import namedtuple
import os

from greg import (
    correlation, force_doubly_nonnegative, decay_model, EMI, covariance_matrix,
    valid_G, hadreg, hadcreg, enforce_directory, load_object, save_object,
    circular_accuracy)

SimCG0 = namedtuple('SimCG0', ['C_obs', 'G0'])


def accuracy_scenario(hadreglparam, data, complex_reg=False):
    if hadreglparam is not None:
        alpha, nu = expit(hadreglparam[0]), expit(hadreglparam[1])
    acc = []
    for simCG0 in data:
        if hadreglparam is not None:
            if not complex_reg:
                G = hadreg(simCG0.G0, alpha=alpha, nu=nu)
                C = simCG0.C_obs
            else:
                G0 = simCG0.G0
                C = hadcreg(simCG0.C_obs, G=G0, alpha=alpha, nu=nu)
                G = hadreg(G0, alpha=alpha, nu=nu)
        else:
            G = simCG0.G0
            C = simCG0.C_obs
        cphases = EMI(C, G=G, corr=False)
        acc.append(circular_accuracy(cphases))
    return np.mean(acc)


def prepare_data(paramlist, rng=None):
    data = []
    for params in paramlist:
        C_obs = correlation(covariance_matrix(decay_model(rng=rng, **params)))
        G0 = valid_G(C_obs, corr=True)
        data.append(SimCG0(C_obs=C_obs, G0=G0))
    return data


def default_paramlist(
        L=100, R=5000, Ps=(40,), coh_decay_list=None, coh_infty_list=None,
        incoh_bad_list=None):
    params0 = {
        'R': R, 'L': L, 'P': 1}
    if coh_decay_list is None: coh_decay_list = [0.5, 0.9]
    if coh_infty_list is None: coh_infty_list = [0.0, 0.1, 0.3, 0.5, 0.7]
    if incoh_bad_list is None: incoh_bad_list = [None, 0.0]
    paramlist = []
    for coh_decay in coh_decay_list:
        for coh_infty in coh_infty_list:
            for incoh_bad in incoh_bad_list:
                for P in Ps:
                    params = params0.copy()
                    params['P'] = P
                    params_new = {
                        'coh_decay': coh_decay, 'coh_infty': coh_infty,
                        'incoh_bad': incoh_bad}
                    params.update(params_new)
                    paramlist.append(params)
    return paramlist


def optimize_hadreg(
        data, hadreglparam0=None, complex_reg=False, maxiter=20, gtol=1e-8):
    if hadreglparam0 is None:
        hadreglparam0 = np.zeros(2)
    f_noreg = accuracy_scenario(None, data, complex_reg=complex_reg)

    def fun(hadreglparam):
        return accuracy_scenario(hadreglparam, data, complex_reg=complex_reg)

    options = {'maxiter': maxiter, 'gtol': gtol}
    res = minimize(fun, hadreglparam0, method='BFGS', options=options)
    hadregres = {'hadreglparam': res.x, 'f': res.fun, 'f_noreg': f_noreg}
    return hadregres


def calibrate_hadreg(
        pathout, looks, seed=1, R=10000, Ps=(40,), complex_reg=False,
        coh_decay_list=None, coh_infty_list=None, incoh_bad_list=None,
        overwrite=False, njobs=-3, maxiter=20):
    res = {}

    def _calibrate_hadreg(L):
        fnout = os.path.join(pathout, f'{L}.p')
        if overwrite or not os.path.exists(fnout):
            rng = default_rng(seed)
            paramlist = default_paramlist(
                L=L, R=R, Ps=Ps, coh_decay_list=coh_decay_list,
                coh_infty_list=coh_infty_list, incoh_bad_list=incoh_bad_list)
            data = prepare_data(paramlist, rng=rng)
            hadregres = optimize_hadreg(
                data, complex_reg=complex_reg, maxiter=maxiter)
            save_object(hadregres, fnout)
        else:
            hadregres = load_object(fnout)
        return hadregres

        res[L] = hadregres

    from joblib import Parallel, delayed
    res = Parallel(n_jobs=njobs)(delayed(_calibrate_hadreg)(L) for L in looks)
    for jL, L in enumerate(looks):
        print(L)
        print(expit(res[jL]['hadreglparam']), res[jL]['f'], res[jL]['f_noreg'])


def calibrate(path0):
    looks = np.arange(3, 26, 1) ** 2
    Ps = (30, 60, 90)
    R = 10000
    scenarios = {
        'broad': (None, None, None), 'low': ([0.5], [0.0], [None]),
        'high': ([0.9], [0.5], [None])}
    rnames = {True: 'G', False: 'complex'}
    for scenario in ('high',):  # scenarios
        for complex_reg in (True , False):
            pathout = os.path.join(path0, rnames[complex_reg], scenario)
            coh_decay_list, coh_infty_list, incoh_bad_list = scenarios[scenario]
            calibrate_hadreg(
                pathout, looks, Ps=Ps, R=R, coh_decay_list=coh_decay_list,
                coh_infty_list=coh_infty_list, incoh_bad_list=incoh_bad_list,
                complex_reg=complex_reg)


if __name__ == '__main__':
    path0 = '/home2/Work/greg/hadamard'
    calibrate(path0)
