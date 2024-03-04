#!/usr/bin/env python3

import numpy as np
import band_structure as bs
from matplotlib import pyplot as plt
## comentar as 4 linhas abaixo caso nao tenha o LaTeX no matplotlib ###
from matplotlib import rc
plt.style.use('bmh')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{physics} \usepackage{bm}')

THETA = 1.05012     # twist angle (in degrees)

a = 2.46   # monolayer graphene lattice constant (in Angstrons)
d = a/np.sqrt(3)    # monolayer nearest neighbor distance
A1 = np.sqrt(3)/2 * a**2    # monolayer unit cell area (in Angstrom^2)
a1 = a * np.array([ 1/2, np.sqrt(3)/2])  # unrotated 1st lattice vector a1
a2 = a * np.array([-1/2, np.sqrt(3)/2])  # unrotated 2nd lattice vector a2
b1 = 4*np.pi/(np.sqrt(3) * a) * np.array([ np.sqrt(3)/2, 1/2])  # unrotated 1st momentum lattice vector b1
b2 = 4*np.pi/(np.sqrt(3) * a) * np.array([-np.sqrt(3)/2, 1/2])  # unrotated 2nd momentum lattice vector b2
kD = 4*np.pi / (3*a)    # monolayer Dirac point absolute value
K  = kD * np.array([1, 0])   # unrotated monolayer Dirac point K of chirality +1
Kp = -K   # unrotated monolayer Dirac point K' of chirality -1


def R(th):  # rotation matrix
    return np.array([[np.cos(th), -np.sin(th)],
                     [np.sin(th),  np.cos(th)]])

### # to generate rotated layers' hamiltonians, use:
### H_layer1 = H_g(k, -th/2)
### H_layer2 = H_g(k, +th/2)
def H_g(k, th):     # monolayer graphene 2x2 hamiltonian in A, B basis ROTATED by angle th
    t = 2.8     # monolayer hopping t = 2.8 eV. See Castro Neto "https://doi.org/10.1103/RevModPhys.81.109"
    delta1 = R(th) @ ((a1 + a2)/3)
    delta2 = R(th) @ ((-2*a1 + a2)/3)
    delta3 = R(th) @ ((a1 - 2*a2)/3)
    # f(k) = sum_n e^{i k \vdot delta_n}
    fk = np.exp(1j*np.dot(k, delta1)) + np.exp(1j*np.dot(k, delta2)) + np.exp(1j*np.dot(k, delta3))
    return np.array([[0, -t * fk],
                     [-t * fk, 0]])


def gen_hamiltonian(q, th=THETA):
    # "q" : is a momentum inside the mini BZ, such that K1 + q is close to K1 (first layer Dirac point)
    # "th": twist angle (in degrees)
    th = th * np.pi / 180   # convert to radians
    th1 = -th/2     # rotate layer 1 by th1
    th2 = +th/2     # rotate layer 2 by th2

    # tau0 is the twisting center translation vector
    tau0 = np.array([0, 0])     # AB-stacking case
    ### tau0 = np.array([0, d])     # AA-stacking case

    ### # translation vectors with tau0 as starting point
    ### # we rotate layer 1 by -th/2 and layer 2 by +th/2
    ### tau1A = R(th1) @ np.array([0, 0])
    ### tau1B = R(th1) @ np.array([0, 0])
    ### tau2A = R(th2) @ (np.array([0, -d]) + tau0)
    ### tau2B = R(th2) @ (np.array([0, 0]) + tau0)

    # rotated layers' lattice vectors
    # b_lj means "momentum lattice vector j of layer l"
    b11 = R(th1) @ b1
    b12 = R(th1) @ b2
    b21 = R(th2) @ b1
    b22 = R(th2) @ b2

    # rotated layers' Dirac points
    K1 = R(th1) @ K
    K2 = R(th2) @ K

    # g lattice vectors that couple the two layers
    g12 =  b12 + 0
    g13 = -b11 + 0
    g22 =  b22 + 0
    g23 = -b21 + 0
    # qb, qtr, and qtl nearest-neighbors moiré vectors
    qb  = K1 - K2
    qtr = K1 - K2 + g12 - g22
    qtl = K1 - K2 + g13 - g23

    tkD = 0.58  # eV.Angstrom^2. See the Handbook "https://doi.org/10.1002/9781119468455.ch44"
    Omega = A1  # monolayer unit cell area (in Angstrom^2)
    w = tkD / Omega     # hopping energy

    # T hopping matrices. See MacDonald "https://doi.org/10.1073/pnas.1108174108"
    phi = 2*np.pi/3
    Tb  = np.array([[1, 1],
                    [1, 1]])
    Ttr  = np.exp(-1j*np.dot(g12, tau0)) * np.array([[np.exp( 1j * phi), 1],
                                                     [np.exp(-1j * phi), np.exp( 1j * phi)]])
    Ttl  = np.exp(-1j*np.dot(g13, tau0)) * np.array([[np.exp(-1j * phi), 1],
                                                     [np.exp( 1j * phi), np.exp(-1j * phi)]])

    zero2x2 = np.zeros((2,2))
    # 4-band hamiltonian
    H4_tbg = np.block([
        [ H_g(K1+q, th1)  , w*Tb            , w*Ttr            , w*Ttl             ],
        [ np.conj(w*Tb).T , H_g(K2+q+qb,th2), zero2x2          , zero2x2           ],
        [ np.conj(w*Ttr).T, zero2x2         , H_g(K2+q+qtr,th2), zero2x2           ],
        [ np.conj(w*Ttl).T, zero2x2         , zero2x2          , H_g(K2+q+qtl,th2) ]
        ])
    return H4_tbg


def eigvals_tbg4(q):
    H = gen_hamiltonian(q)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals


def main():
    th = THETA
    DeltaK = 2*np.sin(th/2)*kD
    Km = np.array([-np.sqrt(3)/2*DeltaK, -1/2*DeltaK])
    path = [
             bs.k_point(r'$K_m$',      np.array([-np.sqrt(3)/2*DeltaK, -1/2*DeltaK])-Km), # K_m
             bs.k_point(r"$K_m'$",     np.array([-np.sqrt(3)/2*DeltaK,  1/2*DeltaK])-Km), # K_m'
             bs.k_point(r'$\Gamma_m$', np.array([0, 0])-Km),                              # Gamma_m
             bs.k_point(r'$M_m$',      np.array([-np.sqrt(3)/2*DeltaK, 0])-Km),           # M_m
             bs.k_point(r'$K_m$',      np.array([-np.sqrt(3)/2*DeltaK, -1/2*DeltaK])-Km), # K_m
           ]
    bs.plot_bandstructure(path, eigenval_func=eigvals_tbg4, ticks_fontsize=20, n_line=100)
    #ymin = 0; ymax = 4.10
    #plt.ylim(ymin, ymax);
    plt.ylabel(r'energy', fontsize=20)
    plt.savefig("band_struct_test-tbg4.png", dpi=300, format='png', bbox_inches="tight")
    plt.clf()


if __name__ == '__main__':
    main()
