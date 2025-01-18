#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
from numpy.linalg import eigh
import band_structure_color as bs
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{physics} \usepackage{stix}')

from scipy.linalg import block_diag

fontsz = 24

th = 1.05 * np.pi/180   # twist angle, in rad
a = 2.46    # monolayer graphene lattice constant, in Angstrons
kabs = 4 * np.pi / (3*a)    # monolayer Dirac point absolute value, in Angstroms^{-1}
kth = 2 * kabs * np.sin(th/2)   # absolute value of moiré Dirac point K_m, in Angstroms^{-1}

# constantes não utilizadas
a_M = a / (2 * np.sin(th/2))     # moiré lattice vector
lamb = 0.3375 * a_M
vpp = -0.03320 * 1e3 # meV . A (v_*'')

# 1 eV = 1e3 meV
v = -4.303 * 1e3 * 1.3 # meV . A (v_*). There is a mysterious 1.3 factor used in publication according to pytwist.py
vp = 1.622 * 1e3 # meV . A (v_*')
M = 3.697   # meV
g = -24.75  # meV (gamma)
eta = 1

s0 = np.array([[ 1, 0 ],        # Pauli 0 (identity matrix)
               [ 0, 1 ]])
sx = np.array([[ 0, 1 ],        # Pauli x
               [ 1, 0 ]])
sy = np.array([[ 0, -1j ],      # Pauli y
               [ 1j,  0 ]])
sz = np.array([[ 1,  0 ],       # Pauli z
               [ 0, -1 ]])
z2 = np.array([[ 0, 0 ],        # 0_{2x2}
               [ 0, 0 ]])

def latex_float(f):
    float_str = "{0:.4g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def R(th):
    # rotation matrix
    r = np.array([
        [np.cos(th),-np.sin(th)],
        [np.sin(th),np.cos(th)]
        ])
    return r

def eigenval_thf_NG(k):       # generate eigenvalues for Topological Heavy Fermion (THF)
    b_moire1 = np.sqrt(3) * kth * np.array([1, 0])
    b_moire2 = R(np.pi/3) @ b_moire1
    cut = 0
    G = [n1*b_moire1+n2*b_moire2 for n1 in range(-10,10) for n2 in range(-10,10) if norm(n1 * b_moire1 + n2 * b_moire2) <= np.sqrt(3) * kth * cut]

    def Hc(q):
        qx = q[0]
        qy = q[1]
        Hc = np.block([[ z2,                          v*(eta*qx*s0  +  1j*qy*sz) ],     # 4x4 matrix
                       [ v*(eta*qx*s0  -  1j*qy*sz),               M*sx          ]])
        return Hc

    Hc_matrices = []
    for i in range(len(G)):
        Hc_matrices.append(Hc(k+G[i]))
    hc = block_diag(*tuple(Hc_matrices))

    def V(q):
        qx = q[0]
        qy = q[1]
        #return np.exp(-norm(q)**2 * lamb**2/2) * np.block([[g*s0 + vp*(eta*qx*sx + qy*sy), vpp * (eta*qx*sx - qy*sy)]])
        return np.block([[g*s0 + vp*(eta*qx*sx + qy*sy), z2]])

    Hfc_matrices = []
    for i in range(len(G)):
        Hfc_matrices.append(V(k+G[i]))

    hfc = np.block(Hfc_matrices)

    hamil = np.block([[  hc, np.conj(np.transpose(hfc))    ],
                      [ hfc,              z2              ]])

    eigval, eigvec = eigh(hamil)
    return eigval

def main():

    # momentum k is in Angstroms^{-1}
    path = [
             bs.k_point(r'$\text{K}_{m}$',      np.array([np.sqrt(3)/2 * kth,  1/2 * kth])), # K_M
             bs.k_point(r'$\Gamma_{m}$',        np.array([0, 0])),                           # Gamma_M
             bs.k_point(r'$\text{M}_{m}$',      np.array([np.sqrt(3)/2 * kth, 0])),          # M_M
             bs.k_point(r"$K_{m}'$",            np.array([np.sqrt(3)/2 * kth, -1/2 * kth])), # K_M'
             #bs.k_point(r'$\text{K}_{m}$',      np.array([np.sqrt(3)/2 * kth,  1/2 * kth])), # K_M
           ]

    bs.plot_bandstructure(path, eigenval_func=eigenval_thf_NG, ticks_fontsize=fontsz, n_line=100, color_index=1)
    ## #plt.plot([], [], ' ', label=r"$v_* = %s \,\text{eV}\cdot\text{\AA}$" % (v*1e-3))
    ## #plt.plot([], [], ' ', label=r"$v_*' = %s \,\text{eV}\cdot\text{\AA}$" % (vp*1e-3))
    ## #plt.plot([], [], ' ', label=r"$M = %s \,\text{meV}$" % (M))
    ## #plt.plot([], [], ' ', label=r"$\gamma = %s \,\text{meV}$" % (g))
    ## #plt.legend()
    ## ymin = -70; ymax = 70   # -70 to 70 in meV
    ## #plt.title(r"$v_* = %s \,\text{eV}\cdot\mathrm{\AA},\; v_*' = %s \,\text{eV}\cdot\mathrm{\AA},\; M = %s \,\text{meV},\; \gamma = %s \,\text{meV}$" % tuple([latex_float(f) for f in [v*1e-3, vp*1e-3, M, g]]), fontsize=fontsz)
    ## plt.title("No mass " + r"$M$" + " and gap " + r"$\gamma$", fontsize=fontsz)
    ## plt.ylim(ymin, ymax);
    ## plt.ylabel(r'Energy (meV)', fontsize=fontsz)
    ## plt.yticks(fontsize=fontsz)
    ## plt.savefig("thf_bands.png", dpi=300, format='png', bbox_inches="tight")
    ## plt.clf()


if __name__ == '__main__':
    main()


#def eigenval_thf_N1(k):       # generate eigenvalues for Topological Heavy Fermion (THF)
#    kx = k[0]
#    ky = k[1]
#
#
#    Hc = np.block([[ z2,                          v*(eta*kx*s0  +  1j*ky*sz) ],     # 4x4 matrix
#                   [ v*(eta*kx*s0  -  1j*ky*sz),               M*sx          ]])
#
#    Hf = 0 * z2     # 2x2 matrix (null matrix)
#
#    #Hfc = np.exp(-norm(k)**2 * lamb**2 / 2) * np.block([[g*s0 + vp*(eta*kx*sx + ky*sy), z2]])   # 2x4 matrix
#    #Hfc = np.block([[g*s0 + vp*(eta*kx*sx + ky*sy), vpp * (eta*kx*sx - ky*sy)]])   # 2x4 matrix
#    Hfc = np.block([[g*s0 + vp*(eta*kx*sx + ky*sy), z2]])   # 2x4 matrix
#
#    hamil = np.block([[  Hc, np.conj(np.transpose(Hfc))    ],
#                      [ Hfc,              Hf              ]])
#    hamil = np.transpose(hamil)
#
#    eigval, eigvec = eigh(hamil)
#    return eigval
