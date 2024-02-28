#!/usr/bin/env python3

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

	def __init__(self, theta, # twist angle
		a = 0.246, # graphene parameters
		vf = 1, u = 0.11, up = 0.11, cut=4): # continuum model parameters

		# convert angles from degrees to radians
		theta = theta*np.pi/180
		self.theta = theta # twist angle

		# graphene parameters
		self.a = a # lattice constant

		# continuum model parameters
		self.v = vf*2.1354*a #vf = 1.3 used in publications
		self.v3 = np.sqrt(3)*a*0.32/2
		self.v4 = np.sqrt(3)*a*0.044/2
		self.gamma1 = 0.4
		self.Dp = 0.05
		self.omega = np.exp(1j*2*np.pi/3)   # e^{i \phi}, \phi = 2 \pi / 3
		self.u = u
		self.up = up

		# define the graphene lattice in momentum space
		k_d = 4*np.pi/(3*a)  # absolute value of monolayer K point
		k1 = np.array([k_d,0])  # 1st K point
		k2 = np.array([np.cos(2*np.pi/3)*k_d,np.sin(2*np.pi/3)*k_d])  # 2nd K point
		k3 = -np.array([np.cos(np.pi/3)*k_d,np.sin(np.pi/3)*k_d])  # 3rd K point

		# generate the moire reciprocal lattice vectors
		q1 = R(theta) @ k1
		q2 = R(theta) @ k2
		q3 = R(theta) @ k3
		q = np.array([q1,q2,q3]) # put them all in a single array
		self.q = q
		k_theta = np.max([norm(q1),norm(q2),norm(q3)]) # used to define the momentum space cutoff
		self.k_theta = k_theta

		# "the Q lattice" refers to a lattice of monolayer K points on which the continuum model is defined
		# basis vectors for the Q lattice
		b1 = q[1]-q[2]
		b2 = q[0]-q[2]
		b3 = q[1]-q[0]
		b = np.array([b1,b2,b3])
		self.b = b

		# generate the Q lattice
        # TESTAR ESSE list[.,.,.] @ array
		Q = np.array([np.array(list([i,j,0]@b - l*q[0]) + [l]) for i in range(-100,100) for j in range(-100,100) for l in [0,1] if norm([i,j,0]@b - l*q[0]) <= np.sqrt(3)*k_theta*cut]) ############ ENTENDER #############
		self.Q = Q
		Nq = len(Q)
		self.Nq = Nq

		# nearest neighbors on the Q lattice
		self.Q_nn={}
		for i in range(Nq):
            # TENTAR ENTENDER ESSA LINHA ABAIXO
			self.Q_nn[i] = [[np.round(Q[:,:2],3).tolist().index(list(np.round(Q[i,:2]+q[j],3))),j] for j in range(len(q)) if list(np.round(Q[i,:2]+q[j],3)) in np.round(Q[:,:2],3).tolist()] ############ ENTENDER #############


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

		>>> tbg = TTGModel(1.05)
		>>> ham =  tbg.gen_ham(0,0)
		>>> vals, vecs = eigh(ham)
		'''

		k = np.array([kx,ky]) # 2d momentum vector

		# create moire hopping matrices for valley index xi
		U1 = np.array((
			[self.u,self.up],
			[self.up,self.u]))

		U2 = np.array((
			[self.u,self.up*self.omega**(-xi)],
			[self.up*self.omega**(xi),self.u]))

		U3 = np.array((
			[self.u,self.up*self.omega**(xi)],
			[self.up*self.omega**(-xi),self.u]))

		# create and populate Hamiltonian matrix
		ham = np.matrix(np.zeros((2*self.Nq,2*self.Nq),dtype=complex))

		for i in range(self.Nq):
			t = self.Q[i,2]
			l = np.sign(2*t-1)
			M = Et(l*xi*self.theta/2,self.phi,l*xi*self.epsilon/2,self.delta)
			E = (M + M.T)/2
			exx = E[0,0]
			eyy = E[1,1]
			exy = E[0,1]

			kj = (I+M)@(k + self.Q[i,:2] + xi*self.A*np.array([exx - eyy, -2*exy]))

			km = xi*kj[0] - 1j*kj[1]


			#Populate diagonal blocks
			ham[2*i,2*i+1] = -self.v*km

			#Populate off-diagonal blocks                                               ###################################
			nn = self.Q_nn[i]                                                           ############ ENTENDER #############
			for neighbor in nn:                                                         ###################################
				j = neighbor[0]
				p = neighbor[1]
				ham[2*i:2*i+2,2*j:2*j+2] = (p==0)*U1 + (p==1)*U2 + (p==2)*U3

		return ham + ham.H

