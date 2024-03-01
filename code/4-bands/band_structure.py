#!/usr/bin/env python3

import numpy as np

from matplotlib import pyplot as plt
## comentar as 4 linhas abaixo caso nao tenha o LaTeX no matplotlib ###
from matplotlib import rc
plt.style.use('bmh')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{physics} \usepackage{bm}')
#matplotlib.verbose.level = 'debug-annoying'
#######################################################################
#from scipy.integrate import simpson

hbar = 1
m = 0.5

def energy_k(trace, dispersion, n_kpoints=600, E_f=0):
    energies = []
    n_trace = int(n_kpoints / (len(trace) - 1))
    for path_num in range(len(trace) - 1):
        for i in range(n_trace):
            z = i / n_trace
            k = trace[path_num] * (1-z) + z * trace[path_num + 1]
            energies.append(dispersion(k))
    energies = np.array(energies)
    energies -= E_f
    return energies

def plot_band(trace, energy):
    n_kpoints = len(energy)
    plt.plot(range(n_kpoints), energy)

def main():
    n_kpoints = 600
    dispersion = lambda k: hbar**2 * np.linalg.norm(k)**2 / (2 * m)
    label_ticks = [r'$\Gamma$', r'$X$', r'$L$', r'$\Gamma$']


    trace_1 = np.array([[0, 0],       # Gamma
                        [1, 0],       # X
                        [1, 1],       # L
                        [0, 0]])      # Gamma
    energy = energy_k(trace_1, dispersion, n_kpoints)
    plot_band(trace_1, energy)

    trace_2 = np.array([[-2, 0],    # Gamma - b1
                        [-1, 0],    # X - b1
                        [-1, 1],    # L - b1
                        [-2, 0]])   # Gamma - b1
    energy = energy_k(trace_2, dispersion, n_kpoints)
    plot_band(trace_2, energy)

    trace_3 = np.array([[0, -2],     # Gamma - b2
                        [1, -2],     # X - b2
                        [1, -1],     # L - b2
                        [0, -2]])    # Gamma - b2
    energy = energy_k(trace_3, dispersion, n_kpoints)
    plot_band(trace_3, energy)


    n_trace = int(n_kpoints / (len(label_ticks)-1))
    normal_ticks = [i*n_trace for i in range(len(label_ticks))]

    #plt.axhline(y=0, ls='--', color='k')
    plt.xlim(normal_ticks[0], normal_ticks[-1])
    plt.xticks(normal_ticks, label_ticks, fontsize=20)
    plt.grid(True, axis='x')

    ymin = 0; ymax = 5.10
    plt.ylim(ymin, ymax);
    plt.ylabel(r'energy$/\mathcal{E}$', fontsize=20)
    plt.savefig("band_struct_square_free.png", dpi=300, format='png', bbox_inches="tight")

if __name__ == '__main__':
    main()
