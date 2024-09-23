import numpy as np
from numpy.linalg import eigh
import band_structure as bs
from matplotlib import pyplot as plt

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


def eigenval_thf(k):       # generate eigenvalues for Topological Heavy Fermion (THF)
    # 1 eV = 1e3 meV
    v = -4.303 * 1e3 # meV . A (v_*)
    #vp = 1.622 * 1e3 # meV . A (v_*')
    vp = 0 # meV . A (v_*')
    M = 3.697   # meV
    #M = 0   # meV
    #g = -24.75  # meV (gamma)
    g = 0  # meV (gamma)
    eta = 1

    kx = k[0]
    ky = k[1]

    Hc = np.block([[ z2,                          v*(eta*kx*s0  +  1j*ky*sz) ],     # 4x4 matrix
                   [ v*(eta*kx*s0  -  1j*ky*sz),               M*sx          ]])

    Hf = 1 * s0     # 2x2 matrix (null matrix)

    Hfc = np.block([[g*s0 + vp*(eta*kx*sx + ky*sy), z2]])   # 2x4 matrix

    H = np.block([[  Hc, np.conj(np.transpose(Hfc)) ],
                  [ Hfc,            Hf              ]])

    eigval, eigvec = eigh(H)
    return eigval


def main():
    th = 1.05 * np.pi/180   # twist angle, in rad
    a = 2.46    # monolayer graphene lattice constant, in Angstrons
    kabs = 4 * np.pi / (3*a)    # monolayer Dirac point absolute value, in Angstroms^{-1}
    kth = 2 * kabs * np.sin(th/2)   # absolute value of moiré Dirac point K_m, in Angstroms^{-1}

    # momentum k is in Angstroms^{-1}
    path = [
             bs.k_point(r'$\text{K}_{\text{M}}$',      np.array([np.sqrt(3)/2 * kth,  1/2 * kth])), # K_m
             bs.k_point(r'$\Gamma_{\text{M}}$', np.array([0, 0])),                           # Gamma_m
             #bs.k_point(r"$K_{\text{M}}'$",     np.array([np.sqrt(3)/2 * kth, -1/2 * kth])), # K_m'
             bs.k_point(r'$\text{M}_{\text{M}}$',      np.array([np.sqrt(3)/2 * kth, 0])),          # M_m
             bs.k_point(r'$\text{K}_{\text{M}}$',      np.array([np.sqrt(3)/2 * kth,  1/2 * kth])), # K_m
           ]

    bs.plot_bandstructure(path, eigenval_func=eigenval_thf, ticks_fontsize=20, n_line=100)
    ymin = -70; ymax = 70   # -70 to 70 in meV
    plt.ylim(ymin, ymax);
    plt.ylabel(r'Energy (meV)', fontsize=20)
    plt.savefig("band_struct_test-tbg.png", dpi=300, format='png', bbox_inches="tight")
    plt.clf()


if __name__ == '__main__':
    main()