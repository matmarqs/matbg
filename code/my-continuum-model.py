#!/usr/bin/env python3

import numpy as np
from numpy.linalg import eigh
from numpy.linalg import norm
import band_structure_color as bs
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{physics} \usepackage{stix}')

from thf_6_bands import main as thf_main

fontsz = 24
colors = bs.colors

def R(th):
    # rotation matrix
    r = np.array([
        [np.cos(th),-np.sin(th)],
        [np.sin(th),np.cos(th)]
        ])
    return r

# 2x2 identity matrix
I = np.identity(2)

class TBGModel():
    '''
    This is the model class for twisted bilayer graphene.
    See Bistritzer and MacDonald PNAS 108 (30) 12233-12237.

    Parameters
    ----------
    theta : float
        Twist angle between the top and bottom bilayer in degrees.
    a : float
        Lattice constant of graphene in nanometers.
    vf : float
        Fermi velocity renormalization constant.  Typical values are 1.2 - 1.3.
    u, up : float
        Interlayer hopping amplitudes in electron-Volts for AA and AB sites respectively.
        u != up captures the effect of finite out of plane corrugation of the moire lattice.
    cut : float
        Momentum space cut off in units of the moire lattice vector.
        Larger values will result in a larger Hamiltonian matrix.
        Convergence is typically attained by cut = 4.

    Examples
    --------
    Construct a model object representing a twisted bilayer graphene device with a twist angle of 1.05 degrees

    >>> tbg = TBGModel(1.05)

    '''

    def __init__(self, theta, # twist angle (in degrees)
        a = 2.46,   # in Angstroms
        vf = 5944,  # in meV . Angstrom
        ratio_w0_w1 = 0.8,  # ratio w0/w1
        w1 = 110,   # in meV
        cut=5):

        # convert angles from degrees to radians
        theta = theta*np.pi/180
        self.theta = theta # twist angle

        # graphene parameters
        self.a = a # lattice constant

        # continuum model parameters
        self.v = vf     # Fermi velocity. There is a mysterious factor 1.3 used in publications
        self.omega = np.exp(1j*2*np.pi/3)   # e^{i \phi}, \phi = 2 \pi / 3
        self.w0 = ratio_w0_w1 * w1
        self.w1 = w1

        # define the graphene lattice in momentum space
        k_d = 4*np.pi/(3*a)  # absolute value of monolayer K point (in Angstrom^{-1})
        self.k_d = k_d
        k1 = np.array([k_d,0])                                          # 1st K point
        k2 = np.array([np.cos(2*np.pi/3)*k_d,np.sin(2*np.pi/3)*k_d])    # 2nd K point C_{3z}(K)
        k3 = np.array([np.cos(4*np.pi/3)*k_d,np.sin(4*np.pi/3)*k_d])    # 3rd K point C_{3z}^2(K)

        # generate the moire reciprocal lattice vectors
        k_theta = k_d * 2 * np.sin(theta/2)
        q1 = k_theta * np.array([0, -1])
        q2 = k_theta * np.array([ np.sqrt(3)/2, 1/2])
        q3 = k_theta * np.array([-np.sqrt(3)/2, 1/2])
        q = np.array([q1,q2,q3]) # put them all in a single array
        self.q = q
        self.k_theta = k_theta

        # "the Q lattice" refers to a lattice of monolayer K points on which the continuum model is defined
        # basis vectors for the Q lattice
        b1 = q[0] - q[1]
        b2 = q[1] - q[2]
        b3 = q[2] - q[0]
        b = np.array([b1,b2,b3])
        self.b = b

        # generate the Q lattice
        Q = np.array([np.array(list([i,j,0]@b + (2*l-1)*q[1]) + [l]) for i in range(-100,100) for j in range(-100,100) for l in [0,1] if norm([i,j,0]@b + (2*l-1)*q[1]) <= np.sqrt(3)*k_theta*cut]) # ENTENDIDO!
        ####Q = np.array([np.array(list([i,j,0]@b + l*q[1]) + [l]) for i in range(-100,100) for j in range(-100,100) for l in [0,1] if norm([i,j,0]@b + l*q[1]) <= np.sqrt(3)*k_theta*cut]) # ENTENDIDO!
        ### EXPLICAÇÃO DA LINHA ACIMA:
        # '''np.array(list([i,j,0]@b - l*q[0]) + [l])''' significa o array [u_x, u_y, 0 ou 1] onde u = i*b1 + j*b2 - l*q1
        # for i in range(-100,100) for j in range(-100,100) for l in [0,1]
        # l = 0 para Q_A e l = 1 para Q_B, por exemplo
        # ''if norm(u) <= np.sqrt(3)*k_theta*cut'' é para que o vetor u não saia do raio delimitado, onde cut = 4 por padrão
        self.Q = Q
        Nq = len(Q)
        self.Nq = Nq    # numero de pontos no Q lattice

        # nearest neighbors on the Q lattice
        self.Q_nn={}
        for i in range(Nq):
            self.Q_nn[i] = [[np.round(Q[:,:2],6).tolist().index(list(np.round(Q[i,:2]+q[j],6))),j] for j in range(len(q)) if list(np.round(Q[i,:2]+q[j],6)) in np.round(Q[:,:2],6).tolist()] # ENTENDIDO!
        ### EXPLICAÇÃO DA LINHA ACIMA:
        # Q[:,:2] retira a terceira componente (0 ou 1) do array Q
        # np.round(Q[:,:2],3) arredonda os float para 3 casas decimais. É feito isso para testar igualdade de floats.
        # np.round(Q[:,:2],3).tolist() transforma o np.array numa lista, para consequentemente aplicar o método .index()
        # .index() é usado em "np.round(Q[:,:2],3).tolist().index()" para procurar um certo Q_i no Q-lattice
        # o argumento de .index() é list(np.round(Q[i,:2]+q[j],3)). Essencialmente é o vetor Q_i + q_j. Ele foi transformado numa lista por list() e arredondado por np.round( ,3)
        # Note que o indexador 'i' é a key do dicionário Q_nn
        # Em resumo, fixado 'i', estamos procurando no Q-lattice os índices dos vetores Q_i + q_j para j = 0,1,2. Retornamos então [ [index(Q_i + q_1), 1], [index(Q_i + q_2), 2], [index(Q_i + q_3), 3] ]
        # Mas só fazemos isso para os vetores Q_i que possuem nearest-neighbors com q1, q2, q3 (black circles)
        # Metade dos outros vetores no Q-lattice possuem nearest-neighbors com -q1, -q2, -q3 (red circles)
        # Para fazer essa verificação, utilizamos a condicional ''if list(np.round(Q[i,:2]+q[j],3)) in np.round(Q[:,:2],3).tolist()''


    #A function to create the hamiltonian for a given point kx, ky
    def gen_ham(self,kx,ky,zeta=1):
        '''
        Generate hamiltonian for a given k-point.

        Parameters
        ----------
        kx, ky : float
            x and y coordinates of the momentum point for which the Hamiltonian is to be generated
            in inverse nanometers.  Note that kx points along Gamma-Gamma of the moire Brillouin zone.
        zeta : +/- 1
            Valley index.

        Returns
        -------
        ham : numpy matrix, shape (2*Nq, 2*Nq), dtype = complex

        Examples
        --------
        Generate Hamiltonian and solve for eigenstates at the K point of the moire Brillouin zone (kx=ky=0).
        Eigenvalues and eigenvectors are stored in vals and vecs respectively.

        >>> tbg = TTGModel(1.05)
        >>> ham =  tbg.gen_ham(0,0)
        >>> vals, vecs = eigh(ham)
        '''

        k = np.array([kx,ky]) # 2d momentum vector

        # create moire hopping matrices for valley index zeta
        U1 = np.array((
            [self.w0,self.w1],
            [self.w1,self.w0]))

        U2 = np.array((
            [self.w0,self.w1*self.omega**(-zeta)],
            [self.w1*self.omega**(zeta),self.w0]))

        U3 = np.array((
            [self.w0,self.w1*self.omega**(zeta)],
            [self.w1*self.omega**(-zeta),self.w0]))

        # create and populate Hamiltonian matrix
        ham = np.matrix(np.zeros((2*self.Nq,2*self.Nq),dtype=complex))  # 2Nq x 2Nq porque temos os sublattices A,B

        for i in range(self.Nq):
            t = self.Q[i,2]     # a terceira componente do array Q, ou seja, 0 ou 1. É o que indexa red ou black circle.
            l = np.sign(2*t-1)  # ao invés de ser 0 ou 1, é -1 ou 1, respectivamente. SIGNIFICA A LAYER -1 (bottom) ou 1 (top)
            ### A linha abaixo estava no código original do https://github.com/sturk111/pytwist
            # M = Et(l*zeta*self.theta/2, self.phi, l*zeta*self.epsilon/2, self.delta)
            ## só que para mim, epsilon = 0, phi = 0, delta = 0 (No caso epsilon = 0 implica delta = 0)
            ## I+M é a matriz de rotação com base na layer l = -1,1 e no valley 'zeta'
            # M = Et(l*zeta*self.theta/2, 0, 0, 0) # = [ [ 0, -th/2 ], [ th/2, 0 ] ]
            M = R(l * zeta * self.theta/2)

            #kj = (I+M)@(k + self.Q[i,:2] + zeta*self.A*np.array([exx - eyy, -2*exy]))
            # zeta*self.A*np.array([exx - eyy, -2*exy]) == 0 no nosso caso
            #kj = M @ (k + self.Q[i,:2]) # literalmente R(th) @ (k + Q_i) ## ACHO QUE ISSO AQUI TA ERRADO (era do pytwist)
            kj = k + self.Q[i,:2]   # AQUI SEM O R(th), que não sei porque deveria ter. Isso aqui dá um plot mais consistente.
            km = zeta*kj[0] - 1j*kj[1]  # zeta * kj_x - 1j * kj_y


            #Populate diagonal blocks
            ham[2*i,2*i+1] = -self.v * km

            #Populate off-diagonal blocks                                               ###################################
            nn = self.Q_nn[i]                                                           ############ ENTENDER #############
                                                                                        ###################################
            for neighbor in nn:     # 'nn' é uma lista que pode estar vazia ou ter 3 elementos da forma [index(Q_i + q_j), j]
                # ou seja, nn é uma lista de três neighbors
                # neighbor é uma lista [j, p]
                j = neighbor[0] # j é o índice do neighbor no Q-lattice
                p = neighbor[1] # p indexa (p == 0: q_b), (p == 1: q_tr), (p == 2: q_tl)
                ham[2*i:2*i+2,2*j:2*j+2] = (p==0)*U1 + (p==1)*U2 + (p==2)*U3
                # ENTENDIDO!

        return ham + ham.H      # essa parte deve dar conta dos delta_{Q'-Q, q_j} + delta_{Q-Q', q_j}

def eigfunc(k):
    kx = k[0]
    ky = k[0]
    hamiltonian = tbg.gen_ham(kx, ky, zeta=1)
    eigval, eigvec = eigh(hamiltonian)
    return eigval

def eigfunc_other(k):
    kx = k[0]
    ky = k[0]
    hamiltonian = tbg.gen_ham(kx, ky, zeta=-1)
    eigval, eigvec = eigh(hamiltonian)
    return eigval

def main():
    a = tbg.a         # monolayer graphene lattice constant, in nanometers
    th = tbg.theta
    kabs = tbg.k_d    # monolayer Dirac point absolute value, in nanometer^{-1}
    kth = tbg.k_theta # absolute value of moiré Dirac point K_m, in nanometer^{-1}

    # momentum k is in Angstroms^{-1}
    path = [
             bs.k_point(r'$K_{m}$',      np.array([np.sqrt(3)/2 * kth,  1/2 * kth])), # K_M
             bs.k_point(r'$\Gamma_{m}$',        np.array([0, 0])),                           # Gamma_M
             #bs.k_point(r"$K_{m}'$",            np.array([np.sqrt(3)/2 * kth, -1/2 * kth])), # K_M'
             bs.k_point(r'$M_{m}$',      np.array([np.sqrt(3)/2 * kth, 0])),          # M_M
             bs.k_point(r'$K_{m}$',      np.array([np.sqrt(3)/2 * kth,  1/2 * kth])), # K_M
           ]

    bs.plot_bandstructure(path, eigenval_func=eigfunc, ticks_fontsize=fontsz, n_line=100, color_index=0)
    #bs.plot_bandstructure(path, eigenval_func=eigfunc_other, ticks_fontsize=fontsz, n_line=100, color_index=1)
    plt.plot(-1000, -1000, color=colors[0], label=r'valley $\zeta = +1$')
    #plt.plot(-1000, -1000, color=colors[1], label=r'valley $\zeta = -1$')
    #plt.plot(-1000, -1000, color=colors[1], label='THF')
    #thf_main()  # Plot from THF code
    ### as linhas abaixos são para colocar legenda com os parâmetros
    ### plt.plot([], [], ' ', label=r"$v_* = %s \,\text{eV}\cdot\text{\AA}$" % (v*1e-3))
    ### plt.plot([], [], ' ', label=r"$v_*' = %s \,\text{eV}\cdot\text{\AA}$" % (vp*1e-3))
    ### plt.plot([], [], ' ', label=r"$M = %s \,\text{meV}$" % (M))
    ### plt.plot([], [], ' ', label=r"$\gamma = %s \,\text{meV}$" % (g))
    plt.legend()
    ymin = -70; ymax = 70   # -ymin to +ymax in meV
    #plt.title(r"$v_* = %s \,\text{eV}\cdot\mathrm{\AA},\; v_*' = %s \,\text{eV}\cdot\mathrm{\AA},\; M = %s \,\text{meV},\; \gamma = %s \,\text{meV}$" % tuple([latex_float(f) for f in [v*1e-3, vp*1e-3, M, g]]), fontsize=fontsz)
    #plt.title("No mass " + r"$M$" + " and gap " + r"$\gamma$", fontsize=fontsz)
    plt.ylim(ymin, ymax);
    plt.ylabel(r'Energy (meV)', fontsize=fontsz)
    plt.yticks(fontsize=fontsz)
    plt.savefig("continuum_model_bands.png", dpi=300, format='png', bbox_inches="tight")
    plt.clf()

if __name__ == '__main__':
    tbg = TBGModel(theta=1.05)  # theta argument is in degrees
    main()
