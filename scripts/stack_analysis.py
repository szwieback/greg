'''
Created on Mar 28, 2022

@author: simon
'''
import os
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import colorcet as cc

from plotting import prepare_figure, colsbg

def read_stack(path0, rtype, imtype='phases'):
    return np.load(os.path.join(path0, rtype, f'{imtype}.npy'))

def cp(_p, mask=None):
    _cp = np.exp(1j * _p)
    if mask is not None:
        _cp[mask] = np.nan
    return _cp

def compare_cphases(p, pr, comptype='all'):
    from greg import circular_accuracy
    axes = {'all': None, 'time': (0, 1), 'space': 2}

    return circular_accuracy(cp(p), cp(pr), axis=axes[comptype])

def local_accuracy(phases, sigma=3):
    from scipy.ndimage import gaussian_filter
    def cgf(cphases, sigma):
        cphases[np.isnan(cphases)] = 0
        m = gaussian_filter(cphases.real, sigma) + 1j * gaussian_filter(cphases.imag, sigma)
        return m
    cphases = cp(phases)
    cphases_f = cgf(cphases, sigma)
    cphases_f /= np.abs(cphases_f)
    acc = gaussian_filter(1 - np.real(cphases * cphases_f.conj()), sigma)
    return acc

    # 32, 12
def plot_interferogram(
        ax, phase, error=None, phase_shift=0.0, cmap='colorwheel', max_error=36,
        min_error=12, vmin=-np.pi, vmax=np.pi):
    import colorcet as cc
    _cmap = cc.cm[cmap]
    _phase = np.angle(np.exp(1j * (phase + phase_shift)))
    vals = _cmap((_phase - vmin) / (vmax - vmin))
    if error is not None:
        _error = (error - min_error) / (max_error - min_error)
        _error[_error > 1] = 1
        _error[_error < 0] = 0
        vals[..., 0:3] *= (1 - _error)[..., np.newaxis]
    him = ax.imshow(vals, cmap=_cmap, vmin=vmin, vmax=vmax)
    return him

def temp_Mongolia():
    path0 = '/home/simon/Work/greg/Mongolia/'
    rtypes = ['none', 'hadamard', 'spectral', 'hadspec']
    phases = {rtype: read_stack(path0, rtype) for rtype in rtypes}
#     for rtype1, rtype2 in combinations(rtypes, 2):
#         acc = compare_cphases(phases[rtype1], phases[rtype2], comptype='all')
#         print(rtype1, rtype2, acc) # convert to equivalent phase sd

    rtype = 'hadamard'
    phases = read_stack(path0, rtype)
    phasesn = read_stack(path0, 'none')
    K_diag = read_stack(path0, rtype, imtype='K_diag')
    K_diag_std_deg = np.sqrt(K_diag) * 180 / np.pi
    C_diag = read_stack(path0, rtype, imtype='C_diag')
    ind = 28
    ind2 = -1

#     mask = K_diag_std_deg[..., ind] > 23.0
    import matplotlib.pyplot as plt
    import colorcet as cc
    fig, axs = plt.subplots(ncols=5, nrows=2)
    fig.set_size_inches((9, 4), forward=True)
    axs[0, 0].imshow(
        np.log(np.abs(C_diag[..., ind])), cmap=cc.cm['gray'], vmin=5.7, vmax=10.5)
    axs[1, 0].imshow(
        np.log(np.abs(C_diag[..., ind2])), cmap=cc.cm['gray'], vmin=5.7, vmax=10.5)
    plot_interferogram(
        axs[0, 1], phases[..., ind], error=K_diag_std_deg[..., ind], phase_shift=0)
    plot_interferogram(
        axs[1, 1], phases[..., ind2], error=K_diag_std_deg[..., ind2], phase_shift=0)

    dphases = np.angle(np.exp(1j * (phases[..., ind] - phasesn[..., ind])))
    plot_interferogram(
        axs[0, 2], dphases, error=K_diag_std_deg[..., ind], vmin=-0.1, vmax=0.1, cmap='bjy')
    dphases = np.angle(np.exp(1j * (phases[..., ind2] - phasesn[..., ind2])))
    plot_interferogram(
        axs[1, 2], dphases, error=K_diag_std_deg[..., ind2], vmin=-0.1, vmax=0.1, cmap='bjy')
    sigma = 3
    dacc = (local_accuracy(phases[..., ind2], sigma=sigma)
            -local_accuracy(phasesn[..., ind2], sigma=sigma))
    axs[1, 3].imshow(dacc, cmap=cc.cm['bjy'], vmin=-0.1, vmax=0.1)
    axs[1, 4].imshow(local_accuracy(phases[..., ind2], sigma=sigma))
    print(np.nanpercentile(dacc, [10, 25, 50, 75, 90]))
    print(np.nanpercentile(local_accuracy(phases[..., ind2], sigma=sigma), [10, 25, 50, 75, 90]))
    print(np.nanpercentile(local_accuracy(phasesn[..., ind2], sigma=sigma), [10, 25, 50, 75, 90]))

#     axs[1, 2].imshow(dphases, vmin=-0.3, vmax=0.3, cmap=cc.cm['bjy'])
#     axs[0, 1].imshow(K_diag_std_deg[..., ind], cmap=cc.cm['gray'], vmin=5, vmax=28)

#     axs[1, 0].imshow(np.angle(cp(phases[..., ind].conj() * phasesn[..., ind], mask=None)), interpolation='nearest')
#     axs[1, 1].imshow(acc)
#     print(np.nanpercentile(np.angle(cp(phases[..., ind].conj() * phasesn[..., ind], mask=mask)), [5, 10, 25, 50, 75, 90, 95]))
    plt.show()

def plot_Finnmark(path0, sigma=2):
#     path0 = '/home/simon/Work/greg/stacks/Finnmark'
    # need to add cbars and labels
    from string import ascii_lowercase
    labels = ['intensity', 'interferogram', 'local dispersion $d_l$',
              '$\\delta d_l$: hs $-$ none', '$\\delta$intfo: hs $-$ none']
    rtype = 'hadspec'
    phases = read_stack(path0, rtype)
    phasesn = read_stack(path0, 'none')
    K_diag = read_stack(path0, rtype, imtype='K_diag')
    K_diag_std_deg = np.sqrt(K_diag) * 180 / np.pi
    C_diag = read_stack(path0, rtype, imtype='C_diag')
    inds = (21, -1)
    llabels, vllabel = ['05-31', '09-16'], '2022'
    vticks = [[], [-np.pi, np.pi], [0.0, 0.7], [-0.1, 0.0, 0.1], [-0.1, 0.0, 0.1]]
    vticklabels = [[], ['$-\\pi$', '$\\pi$'], ['$0.0$', '$0.7$'], ['$-0.1$', '', '$0.1$'],
                   ['$-0.1$', '', '$0.1$']]
    vunits = ['dB', '-', '-', '-', '-']
    p = lambda im: im
    handles = [[] for ind in inds]
    fig, axs = prepare_figure(
        nrows=2, ncols=5, figsize=(2.0, 0.44), remove_spines=False, left=0.04, right=0.99,
        wspace=0.1, hspace=0.03, bottom=0.20, top=0.93)
    for jr, ind in enumerate(inds):
        handles[jr].append(axs[jr, 0].imshow(
            p(np.log(np.abs(C_diag[..., ind]))), cmap=cc.cm['gray'], vmin=5.7, vmax=10.5))
        handles[jr].append(plot_interferogram(
            axs[jr, 1], p(phases[..., ind]), error=p(K_diag_std_deg[..., ind])))
        handles[jr].append(axs[jr, 2].imshow(
            p(local_accuracy(phases[..., ind], sigma=sigma)), cmap=cc.cm['gray_r'],
            vmin=0.0, vmax=0.7))
        dacc = lambda _ind: (local_accuracy(phases[..., _ind], sigma=sigma)
                            -local_accuracy(phasesn[..., _ind], sigma=sigma))
        handles[jr].append(
            axs[jr, 3].imshow(p(dacc(ind)), cmap=cc.cm['bjy'], vmin=-0.1, vmax=0.1))
        dphases = np.angle(np.exp(1j * (phases[..., ind] - phasesn[..., ind])))
        handles[jr].append(plot_interferogram(
            axs[jr, 4], p(dphases), error=p(K_diag_std_deg[..., ind]),
            vmin=-0.1, vmax=0.1, cmap='bjy'))

    for jc, ax in enumerate(axs[0, :]):
        ax.text(
            0.01, 1.08, f"{ascii_lowercase[jc]}) {labels[jc]}", c='k', va='baseline',
            ha='left', transform=ax.transAxes)
    for jc, ax in enumerate(axs[-1, :]):
        cax = ax.inset_axes([0.22, -0.28, 0.56, 0.18], transform=ax.transAxes)
        cbar = fig.colorbar(handles[-1][jc], ax=ax, cax=cax, orientation='horizontal')
        cbar.ax.set_xticks(vticks[jc])
        cbar.ax.set_xticklabels(vticklabels[jc])
        cbar.solids.set_rasterized(True)
        ax.text(
            1.00, -0.25, f'[{vunits[jc]}]', transform=ax.transAxes, ha='right', 
            va='baseline')

    for jr, ax in enumerate(axs[:, 0]):
        ax.text(
            -0.03, 0.50, llabels[jr], va='center', ha='right', transform=ax.transAxes, 
            rotation=90)
    ax.text(
        -0.12, 1.07, vllabel, va='center', ha='right', transform=ax.transAxes, 
        rotation=90)
    for ax in axs.flatten():
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
#     for _ind in [ind, ind2]:
#         print(np.nanpercentile(dacc(_ind), [10, 25, 50, 75, 90]))
#         print(np.nanpercentile(local_accuracy(phases[..., _ind], sigma=sigma), [10, 25, 50, 75, 90]))
#         print(np.nanpercentile(local_accuracy(phasesn[..., _ind], sigma=sigma), [10, 25, 50, 75, 90]))
#     plt.show()
    plt.savefig(os.path.join(path0, 'figures', 'stack.pdf'), dpi=450)


def prepare_accuracy(path0, sigma=2, percentiles=(5, 25, 50, 75, 95), overwrite=False):
    from greg import save_object, load_object
    rtypes = ['none', 'hadamard', 'spectral', 'hadspec']
    phases = {rtype: read_stack(path0, rtype) for rtype in rtypes}

    fnrelacc = os.path.join(path0, 'figures', 'relacc.p')
    if not os.path.exists(fnrelacc) or overwrite:
        relacc = {}
        for rtype1, rtype2 in combinations(rtypes, 2):
            acc = compare_cphases(phases[rtype1], phases[rtype2], comptype='all')
            relacc[(rtype1, rtype2)] = acc
        save_object(relacc, fnrelacc)
    else:
        relacc = load_object(fnrelacc)

    fndla = os.path.join(path0, 'figures', 'dla.p')
    r0 = rtypes[0]

    if not os.path.exists(fndla) or overwrite:
        dla = {}
        for rtype in [rtype for rtype in rtypes if rtype != r0]:
            dla[rtype] = []
            for ind in range(phases[r0].shape[-1]):
                lan = local_accuracy(phases[r0][..., ind], sigma=sigma).flatten()
                la = local_accuracy(phases[rtype][..., ind], sigma=sigma).flatten()
                dla[rtype].append(np.nanpercentile(la - lan, percentiles))
        save_object(dla, fndla)
    else:
        dla = load_object(fndla)

    return relacc, dla

def plot_summary(path0, vmin=0.0, vmax=0.2, overwrite=False):
    relacc, dla = prepare_accuracy(path0, overwrite=overwrite)
    abbr = {'hadamard': 'h', 'spectral': 's', 'hadspec': 'hs', 'none': 'n'}
    rtypes_plot = ['hadamard', 'spectral', 'hadspec', 'none']
    rtypes_plot2 = ['hadamard', 'spectral', 'hadspec']

    fig, ax = prepare_figure(nrows=1, ncols=1, figsize=(0.38, 0.42), left=0.1, right=0.99,
        wspace=0.1, hspace=0.03, bottom=0.3, top=0.98, sharex=False, sharey=False,
        remove_spines=False)
    acc_m = np.zeros((4, 4), dtype=np.float32)
    for jr1, r1 in enumerate(rtypes_plot):
        for jr2, r2 in enumerate(rtypes_plot):
            if (r1, r2) in relacc:
                acc_m[jr1, jr2] = relacc[(r1, r2)]
            elif (r2, r1) in relacc:
                acc_m[jr1, jr2] = relacc[(r2, r1)]
    mp = ax.imshow(acc_m, cmap=cc.cm['gray_r'], vmin=vmin, vmax=vmax)
    cax = ax.inset_axes([0.05, -0.40, 0.60, 0.12], transform=ax.transAxes)
    cbar = fig.colorbar(mp, ax=ax, cax=cax, orientation='horizontal')
    cbar.ax.set_xticks([])
    cbar.solids.set_rasterized(True)
    cax.text(1.10, 0.03, f'{vmax} [-]', ha='left', va='baseline', transform=cax.transAxes)
    cax.text(-0.10, 0.10, f'{vmin}', ha='right', va='baseline', transform=cax.transAxes)
    ax.set_xticks(list(range(len(rtypes_plot))))
    ax.set_yticks(list(range(len(rtypes_plot))))
    tickl = [abbr[rtype] for rtype in rtypes_plot]
    ax.set_xticklabels(tickl)
    ax.set_yticklabels(tickl)
    plt.savefig(os.path.join(path0, 'figures', 'stackacc.pdf'), dpi=450)

    fig, axs = prepare_figure(nrows=3, ncols=1, figsize=(0.59, 0.42), left=0.30, right=0.98,
        wspace=0.1, hspace=0.2, bottom=0.15, top=0.98, sharey=True, sharex=True)
    xpos = np.arange(len(dla[rtypes_plot2[0]]))
    for jrtype, rtype in enumerate(rtypes_plot2):
        _dla = np.array(dla[rtype])
        axs[jrtype].fill_between(
            xpos, _dla[:, 1], _dla[:, 3], alpha=0.4, facecolor=colsbg[2])
        axs[jrtype].axhline(0.0, lw=0.5, color='#cccccc')
        axs[jrtype].plot(xpos, _dla[:, 2], c=colsbg[0], lw=1.0)
        print(np.min(_dla[:, 1]), np.max(_dla[:, 3]))
        axs[jrtype].text(
            0.02, 0.06, abbr[rtype], transform=axs[jrtype].transAxes,
            ha='left', va='baseline')
    axs[0].set_ylim([-0.07, 0.03])
    axs[1].text(
        -0.34, 0.50, '$\\Delta d_l$ [-]', ha='right', va='center', rotation=90,
        transform=axs[1].transAxes)
    xticks = [1, 10, 19, 28]
    xticklabels = ['Sep', 'Jan', 'May', 'Sep']
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(xticklabels)
    plt.savefig(os.path.join(path0, 'figures', 'stackdeltad.pdf'))

if __name__ == '__main__':
    path0 = '/home/simon/Work/greg/stacks/Finnmark'
#     plot_summary(path0)
    plot_Finnmark(path0)

