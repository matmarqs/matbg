#!/usr/bin/env python3

import numpy as np
import band_structure_color as bs
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{physics} \usepackage{stix}')

fontsz = 24

a = 2.46    # monolayer graphene lattice constant, in Angstrons
kabs = 4 * np.pi / (3*a)    # monolayer Dirac point absolute value, in Angstroms^{-1}
b1 = 4*np.pi/(np.sqrt(3) * a) * np.array([np.sqrt(3)/2, -1/2])
b2 = 4*np.pi/(np.sqrt(3) * a) * np.array([0, 1])
Gamma = np.array([0, 0])
K = (2*b1 + b2)/3
M = (b1 + b2)/2

def eigenval(k):       # generate eigenvalues
    path = "KM"
    if equal_zero(k[1]):
        path = "GK"

    total = np.linalg.norm(K-Gamma) + np.linalg.norm(M-K)

    if path == "GK":
        u = np.linalg.norm(k - Gamma) / total
    elif path == "KM":
        u = (np.linalg.norm(K - Gamma) + np.linalg.norm(k - K)) / total

    u_Gamma = 0
    u_K = np.linalg.norm(K-Gamma) / total
    u_KM2 = (np.linalg.norm(K-Gamma) + np.linalg.norm(M-K)/2) / total
    u_M = 1

    ### connected bands
    #f1 = interp1d([u_Gamma, u_K/2, u_K, u_KM2, u_M], [2, 1.2, 1, 1.5, 2], kind="cubic", fill_value="extrapolate")
    #f2 = interp1d([u_Gamma, u_K/2, u_K, u_KM2, u_M], [2, 2.5, 3, 3.5, 4], kind="cubic", fill_value="extrapolate")
    #f3 = interp1d([u_Gamma, u_K/2, u_K, u_KM2, u_M], [4, 3.5, 3, 2.5, 2], kind="cubic", fill_value="extrapolate")
    #f4 = interp1d([u_Gamma, u_K/2, u_K, u_KM2, u_M], [4, 4.8, 5, 3.5, 4], kind="cubic", fill_value="extrapolate")

    # disconnected bands
    f1 = interp1d([u_Gamma, u_K/2, u_K, u_M], [2, 1.5, 1, 2], kind="cubic", fill_value="extrapolate")
    f2 = interp1d([u_Gamma, u_K/2, u_K, u_M], [2, 2.5, 3, 2], kind="cubic", fill_value="extrapolate")
    f3 = interp1d([u_Gamma, u_K/2, u_K, u_KM2, u_M], [4, 3.5, 4, 4.3, 4], kind="cubic", fill_value="extrapolate")
    f4 = interp1d([u_Gamma, u_K/2, u_K, u_KM2, u_M], [4, 4.5, 4, 3.7, 4], kind="cubic", fill_value="extrapolate")

    eigval = [f1(u), f2(u), f3(u), f4(u)]

    return eigval

def equal_zero(x):
    return np.abs(x < 1e-6)

def main():
    # momentum k is in Angstroms^{-1}
    path = [
             bs.k_point(r'$\Gamma$', Gamma),                                        # Gamma
             bs.k_point(r'$K$',      K),                                            # K
             bs.k_point(r'$M$',      M),                                            # M
             #bs.k_point(r'$K$',      np.array([np.sqrt(3)/2 * kabs,  1/2 * kabs])),# K
           ]

    bs.plot_bandstructure(path, eigenval_func=eigenval, ticks_fontsize=fontsz, n_line=100)
    #plt.title("connected set of bands", fontsize=fontsz)
    plt.title("disconnected set of bands", fontsize=fontsz)
    ###plt.ylim(ymin, ymax);
    ###plt.ylabel(r'Energy (meV)', fontsize=fontsz)
    ###plt.yticks(fontsize=fontsz)
    plt.yticks([])
    #plt.savefig("bandstructure_con.png", dpi=300, format='png', bbox_inches="tight")
    plt.savefig("bandstructure_discon.png", dpi=300, format='png', bbox_inches="tight")
    plt.clf()


if __name__ == '__main__':
    main()

