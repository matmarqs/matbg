# From https://github.com/sturk111/pytwist

import numpy as np
from numpy.linalg import eigh
from numpy.linalg import norm
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

## comentar as 4 linhas abaixo caso nao tenha o LaTeX no matplotlib ###
from matplotlib import rc
plt.style.use('bmh')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{physics} \usepackage{bm} \usepackage{stix}')
#######################################################################

fontsz= 24
colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']

#-----------------------------------------------------------------------------------------------
#Helper Functions

def R(x):
    #Generic rotation matrix
    r = np.array((
        [np.cos(x),-np.sin(x)],
        [np.sin(x),np.cos(x)]
        ))
    return r

#2x2 Identity matrix
I=np.identity(2)

def Et(theta, theta_s, e, delta):
    '''
    A matrix that simultaneously rotates and strains the vector on which it operates.

    See Bi et al. Phys. Rev. B 100, 035448 (2019).

    Parameters
    ----------
    theta : float
        Rotation angle in radians.
    theta_s : float
        Strain angle in radians.
    e : float
        Strain magnitude.
    delta : float
        Poisson ratio.

    Returns
    -------
    result : 2x2 numpy array
    '''
    return R(-theta_s)@np.array(([e,0],[0,-delta*e]))@R(theta_s) + np.array(([0,-theta],[theta,0]))

#-----------------------------------------------------------------------------------------------

class TBGModel():
    '''
    This is the model class for twisted bilayer graphene.

    See Bistritzer and MacDonald PNAS 108 (30) 12233-12237.

    Parameters
    ----------
    theta : float
        Twist angle between the top and bottom bilayer in degrees.
    phi : float
        Heterostrain angle in degrees relative to the atomic lattice.
    epsilon : float
        Heterostrain magnitude.  For example, a value of 0.01 corresponds to a 1%
        difference in lattice constants between the top and bottom layers.
    a : float
        Lattice constant of graphene in nanometers.
    beta : float
        Two center hopping modulus.  See Phys. Rev. B 100, 035448 (2019).
    delta : float
        Poisson ratio for graphene.
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
    Construct a model object representing a twisted bilayer graphene device with a twist angle
    of 1.11 degrees, a strain angle of 15 degrees, a strain magnitude of 0.5%.

    >>> tbg = TBGModel(1.11, 15, 0.005)

    '''
    def __init__(self, theta, phi, epsilon, #Empirical parameters (must provide as input)
        a = 2.46, beta = 3.14, delta = 0.16, #Graphene parameters. beta and delta do not matter
        vf = 5944, u = 88, up = 110, cut=4): #Continuum model parameters

        #Convert angles from degrees to radians
        theta = theta*np.pi/180
        phi = phi*np.pi/180

        #Empirical parameters
        self.theta = theta #twist angle
        self.phi = phi #strain angle
        self.epsilon = epsilon #strain percent

        #Graphene parameters
        self.a = a #lattice constant
        self.beta = beta #two center hopping modulus
        self.delta = delta #poisson ratio
        self.A = np.sqrt(3)*self.beta/2/a #gauge connection

        #Continuum model parameters
        self.v = vf  #vf = 1.3 used in publications
        self.v3 = np.sqrt(3)*a*0.32/2
        self.v4 = np.sqrt(3)*a*0.044/2
        self.gamma1 = 0.4
        self.Dp = 0.05

        self.omega = np.exp(1j*2*np.pi/3)
        self.u = u
        self.up = up

        #Define the graphene lattice in momentum space
        k_d = 4*np.pi/3/a
        k1 = np.array([k_d,0])
        k2 = np.array([np.cos(2*np.pi/3)*k_d,np.sin(2*np.pi/3)*k_d])
        k3 = -np.array([np.cos(np.pi/3)*k_d,np.sin(np.pi/3)*k_d])

        #Generate the strained moire reciprocal lattice vectors
        q1 = Et(theta,phi,epsilon,delta)@k1
        q2 = Et(theta,phi,epsilon,delta)@k2
        q3 = Et(theta,phi,epsilon,delta)@k3
        q = np.array([q1,q2,q3]) #put them all in a single array
        self.q = q
        k_theta = np.max([norm(q1),norm(q2),norm(q3)]) #used to define the momentum space cutoff
        self.k_theta = k_theta

        #"the Q lattice" refers to a lattice of monolayer K points on which the continuum model is defined
        #basis vectors for the Q lattice
        b1 = q[1]-q[2]
        b2 = q[0]-q[2]
        b3 = q[1]-q[0]
        b = np.array([b1,b2,b3])
        self.b = b

        #generate the Q lattice
        ######## Q = np.array([np.array(list([i,j,0]@b - (2*l-1)*q[0]) + [l]) for i in range(-100,100) for j in range(-100,100) for l in [0,1] if norm([i,j,0]@b - (2*l-1)*q[0]) <= np.sqrt(3)*k_theta*cut])
        Q = np.array([np.array(list([i,j,0]@b - l*q[0]) + [l]) for i in range(-100,100) for j in range(-100,100) for l in [0,1] if norm([i,j,0]@b - l*q[0]) <= np.sqrt(3)*k_theta*cut])
        self.Q = Q
        Nq = len(Q)
        self.Nq = Nq

        #nearest neighbors on the Q lattice
        self.Q_nn={}
        for i in range(Nq):
            self.Q_nn[i] = [[np.round(Q[:,:2],3).tolist().index(list(np.round(Q[i,:2]+q[j],3))),j] for j in range(len(q)) if list(np.round(Q[i,:2]+q[j],3)) in np.round(Q[:,:2],3).tolist()]

    #A function to create the hamiltonian for a given point kx, ky
    def gen_ham(self,kx,ky,xi=1):
        '''
        Generate hamiltonian for a given k-point.

        Parameters
        ----------
        kx, ky : float
            x and y coordinates of the momentum point for which the Hamiltonian is to be generated
            in inverse nanometers.  Note that kx points along Gamma-Gamma of the moire Brillouin zone.
        xi : +/- 1
            Valley index.

        Returns
        -------
        ham : numpy matrix, shape (2*Nq, 2*Nq), dtype = complex

        Examples
        --------
        Generate Hamiltonian and solve for eigenstates at the K point of the moire Brillouin zone (kx=ky=0).
        Eigenvalues and eigenvectors are stored in vals and vecs respectively.

        >>> tbg = TTGModel(1.11, 15, 0.005)
        >>> ham =  tbg.gen_ham(0,0)
        >>> vals, vecs = eigh(ham)
        '''
        k = np.array([kx,ky]) #2d momentum vector

        #Create moire hopping matrices for valley index xi
        U1 = np.array((
            [self.u,self.up],
            [self.up,self.u]))

        U2 = np.array((
            [self.u,self.up*self.omega**(-xi)],
            [self.up*self.omega**(xi),self.u]))

        U3 = np.array((
            [self.u,self.up*self.omega**(xi)],
            [self.up*self.omega**(-xi),self.u]))

        #Create and populate Hamiltonian matrix
        ham = np.matrix(np.zeros((2*self.Nq,2*self.Nq),dtype=complex))

        for i in range(self.Nq):
            t = self.Q[i,2]
            l = np.sign(2*t-1)
            ## M = Et(l*xi*self.theta/2,self.phi,l*xi*self.epsilon/2,self.delta)
            M = Et(l*self.theta/2,self.phi,l*xi*self.epsilon/2,self.delta)
            E = (M + M.T)/2
            exx = E[0,0]
            eyy = E[1,1]
            exy = E[0,1]

            ### kj = (I+M)@(k + self.Q[i,:2] + xi*self.A*np.array([exx - eyy, -2*exy]))
            kj = (k + self.Q[i,:2] + xi*self.A*np.array([exx - eyy, -2*exy]))

            km = xi*kj[0] - 1j*kj[1]


            #Populate diagonal blocks
            ham[2*i,2*i+1] = -self.v*km

            #Populate off-diagonal blocks
            nn = self.Q_nn[i]
            for neighbor in nn:
                j = neighbor[0]
                p = neighbor[1]
                ham[2*i:2*i+2,2*j:2*j+2] = (p==0)*U1 + (p==1)*U2 + (p==2)*U3

        return ham + ham.H

    #A function to solve for the bands along the path: K -> Gamma -> M -> K'
    def solve_along_path(self, res=16, plot_it = True, return_eigenvectors = False): #res = number of points per unit length in k space
        '''
        Compute the band structure along a path within the moire Brillouin zone.
        The path is predefined to be K -> Gamma -> M -> K'.

        Parameters
        ----------
        res : float
            The resolution of the cut, defined as the number of points to sample along the line Gamma -> K
        plot_it : boolean
            If true a function call will display a plot of the resulting band structure.
        return_eigenvectors : boolean
            If true return eigenvectors along the k path.

        Returns
        -------
        evals_m : numpy array, shape (len(kpath), 2*Nq)
            Eigenvalues along the k path for valley xi = -1. Axis 0 indexes points along the k path.
            Axis 1 indexes bands.

        evals_p : numpy array, shape (len(kpath), 2*Nq)
            Eigenvalues along the k path for valley xi = +1. Axis 0 indexes points along the k path.
            Axis 1 indexes bands.
        evecs_m : numpy array, shape (len(kpath), 2*Nq, 2*Nq)
            Eigenvectors along the k path for valley xi = -1.  Axis 0 indexes points along the k path.
            Axis 1 indexes into the eigenvector.  Axis 2 indexes bands.  E.g. evecs_m[i,:,j] is the
            eigenvector for k point i and band j.
        evecs_p :numpy array, shape (len(kpath), 2*Nq, 2*Nq)
            Eigenvectors along the k path for valley xi = +1.  Axis 0 indexes points along the k path.
            Axis 1 indexes into the eigenvector.  Axis 2 indexes bands.  E.g. evecs_p[i,:,j] is the
            eigenvector for k point i and band j.
        kpath : list of shape (2,) numpy arrays
            List of k points along the path.  Each point is represented as a two component array in the list.

        Examples
        --------
        Solve for the bands along the path K -> Gamma -> M -> K', returning eigenvectors.

        >>> tbg = TDBGModel(1.11, 15, 0.005)
        >>> evals_m, evals_p, evecs_m, evecs_p, kpath = tbg.solve_along_path(return_eigenvectors = True)

        The same thing without returning eigenvectors.

        >>> evals_m, evals_p, kpath = tbg.solve_along_path()

        A higher resolution computation.

        >>> evals_m, evals_p, kpath = tbg.solve_along_path(res = 64)
        '''

        l1 = int(res) #K->Gamma
        l2 = int(np.sqrt(3)*res/2) #Gamma->M
        l3 = int(res/2) #M->K'

        kpath = [] #K -> Gamma -> M -> K'
        for i in np.linspace(0,1,l1):
            ######## kpath.append((1-i)*(self.q[0]+self.q[2])) #K->Gamma
            kpath.append(i*(self.q[0]+self.q[1])) #K->Gamma
        for i in np.linspace(0,1,l2):
            ######## kpath.append(i*(self.q[0]/2+self.q[2])) #Gamma->M
            kpath.append(self.q[0] + self.q[1] + i*(-self.q[0]/2 - self.q[1])) #Gamma->M
        for i in np.linspace(0,1,l3):
            kpath.append((1-i)*self.q[0]/2) #M->K
            ### kpath.append(self.q[0]/2 + i*self.q[0]/2) #M->K'
            ######## kpath.append((1-i)*(self.q[0]/2+self.q[2]) + i*(self.q[0]+self.q[2])) #M->K

        ######## print("q1:", self.q[0])
        ######## print("q2:", self.q[1])
        ######## print("q3:", self.q[2])
        ######## print("K->Gamma:", self.q[0]+self.q[1])
        ######## print("Gamma->M:", self.q[0] + self.q[1] + 1*(-self.q[0]/2 - self.q[1]))
        ######## print("M->K':", self.q[0]/2 + 1*self.q[0]/2)

        evals_m = []
        evals_p = []
        evals_thf = []
        if return_eigenvectors:
            evecs_m = []
            evecs_p = []

        for kpt in kpath: #for each kpt along the path
            ham_m = self.gen_ham(kpt[0],kpt[1],-1) #generate and solve a hamiltonian for each valley
            ham_p = self.gen_ham(kpt[0],kpt[1],1)

            val, vec = eigh(ham_m)
            evals_m.append(val)
            if return_eigenvectors:
                evecs_m.append(vec)

            val, vec = eigh(ham_p)
            evals_p.append(val)
            if return_eigenvectors:
                evecs_p.append(vec)

            ### THF
            evals_thf.append(eigenval_thf(kpt-(self.q[0]+self.q[1])))

        evals_m = np.array(evals_m)
        evals_p = np.array(evals_p)
        evals_thf = np.array(evals_thf)

        if plot_it:
            plt.figure(1)
            plt.clf()
            for i in range(len(evals_m[1,:])):
                #### plt.plot(evals_m[:,i],linestyle='dashed',color=colors[2])
                plt.plot(evals_p[:,i],color=colors[0])

            for j in range(len(evals_thf[1,:])):
                plt.plot(evals_thf[:,j], color=colors[1])

            plt.ylim(-70,70)
            ### plt.xticks([0,l1,l1+l2,l1+l2+l3],['K', r'$\Gamma$', 'M', 'K\''])
            ### total = l1 + l2 + l3
            plt.xticks([0,l1,l1+l2,l1+l2+l3],[r'$K_m$', r'$\Gamma_m$', '$M_m$', '$K_m$'],fontsize=fontsz)
            plt.xlim(0, l1+l2+l3)
            plt.ylabel('Energy (meV)', fontsize=fontsz)
            plt.yticks(fontsize=fontsz)
            #plt.tight_layout()

        if return_eigenvectors:
            evecs_m = np.array(evecs_m)
            evecs_p = np.array(evecs_p)
            return evals_m, evals_p, evecs_m, evecs_p, kpath

        else:
            return evals_m, evals_p, kpath

def eigenval_thf(k):       # generate eigenvalues for Topological Heavy Fermion (THF)
    # 1 eV = 1e3 meV
    ### v = -4.303 * 1e3 * 1.3 # meV . A (v_*). There is a mysterious 1.3 factor used in publication according to pytwist.py
    v = -4.303 * 1e3 # meV . A (v_*). There is a mysterious 1.3 factor used in publication according to pytwist.py
    vp = 1.622 * 1e3 # meV . A (v_*')
    M = 3.697   # meV
    g = -24.75  # meV (gamma)
    eta = -1

    th = 1.05 * np.pi/180   # twist angle, in rad
    a = 2.46    # monolayer graphene lattice constant, in Angstrons
    kabs = 4 * np.pi / (3*a)    # monolayer Dirac point absolute value, in Angstroms^{-1}
    kth = 2 * kabs * np.sin(th/2)   # absolute value of moiré Dirac point K_m, in Angstroms^{-1}

    # constantes não utilizadas
    a_M = a / (2 * np.sin(th/2))     # moiré lattice vector
    lamb = 0.3375 * a_M
    vpp = -0.03320 * 1e3 # meV . A (v_*'')

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
        return np.exp(-norm(q)**2 * lamb**2/2) * np.block([[g*s0 + vp*(eta*qx*sx + qy*sy), vpp * (eta*qx*sx - qy*sy)]])
        #return np.block([[g*s0 + vp*(eta*qx*sx + qy*sy), z2]])

    Hfc_matrices = []
    for i in range(len(G)):
        Hfc_matrices.append(V(k+G[i]))

    hfc = np.block(Hfc_matrices)

    hamil = np.block([[  hc, np.conj(np.transpose(hfc))    ],
                      [ hfc,              z2              ]])

    eigval, eigvec = eigh(hamil)
    return eigval


tbg = TBGModel(1.05, 0, 0)
evals_m, evals_p, kpath = tbg.solve_along_path(res = 64)

#thf_main()
plt.plot(-1000, -1000, color=colors[0], label=r'BM')
plt.plot(-1000, -1000, color=colors[1], label=r'THF')
plt.legend()
plt.gca().set_aspect(aspect=1.2)

plt.savefig("output_pytwist.png", dpi=300, format='png', bbox_inches="tight")
