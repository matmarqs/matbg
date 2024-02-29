#!/usr/bin/env python3

import numpy as np

a = 0.246   # monolayer graphene lattice constant (in nanometers)
d = a/np.sqrt(3)    # monolayer nearest neighbor distance
A1 = np.sqrt(3)/2 * a**2    # monolayer unit cell area
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

def monolayer_H(k, th):     # monolayer graphene 2x2 hamiltonian in A, B basis ROTATED by angle th
    t = 2.8     # monolayer hopping t = 2.8 eV. see "10.1103/RevModPhys.81.109"
    delta1 = R(th) @ ((a1 + a2)/3)
    delta2 = R(th) @ ((-2*a1 + a2)/3)
    delta3 = R(th) @ ((a1 - 2*a2)/3)
    fk = np.exp(1j*np.dot(k, delta1)) + np.exp(1j*np.dot(k, delta2)) + np.exp(1j*np.dot(k, delta3))     # f(k) = sum_{nu} e^{i k \vdot delta_nu}
    return np.array([[0, -t * fk],
                     [-t * fk, 0]])

def main():
    th = 1.05012   # twist angle

    # tau0 is the twisting center translation vector
    tau0 = np.array([0, 0])     # AB-stacking case
    #tau0 = np.array([0, d])     # AA-stacking case

    # translation vectors with tau0 as starting point
    # we rotate layer 1 by -th/2 and layer 2 by +th/2
    tau1A = R(-th/2) @ np.array([0, 0])
    tau1B = R(-th/2) @ np.array([0, 0])
    tau2A = R(+th/2) @ (np.array([0, -d]) + tau0)
    tau2B = R(+th/2) @ (np.array([0, 0]) + tau0)

    # rotated layers' lattice vectors
    # b_lj means "momentum lattice vector j of layer l"
    b11 = R(-th/2) @ b1
    b12 = R(-th/2) @ b2
    b21 = R(+th/2) @ b1
    b22 = R(+th/2) @ b2

    # rotated layers' Dirac points
    K1 = R(-th/2) @ K
    K2 = R(+th/2) @ K

    # generate rotated layers' hamiltonians
    H1 = monolayer_H(k, -th/2)
    H2 = monolayer_H(k, +th/2)
