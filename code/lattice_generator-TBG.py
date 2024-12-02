#!/usr/bin/env python3

import numpy as np

from matplotlib import pyplot as plt
## comentar as 4 linhas abaixo caso nao tenha o LaTeX no matplotlib ###
from matplotlib import rc
plt.style.use('bmh')
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)
#rc('text.latex', preamble=r'\usepackage{physics} \usepackage{bm}')
#matplotlib.verbose.level = 'debug-annoying'
#######################################################################
#from scipy.integrate import simpson

def rotation(a, th):
    return np.array([np.cos(th) * a[0] - np.sin(th) * a[1], np.sin(th) * a[0] + np.cos(th) * a[1]])

# colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']
def draw_2DLattice(a1, a2, basis, color):
    M = np.linspace(-20, 20, 41)
    N = np.linspace(-20, 20, 41)
    for x in M:
        for y in N:
            Rx = x * a1[0] + y * a2[0]
            Ry = x * a1[1] + y * a2[1]
            for b in basis:
                plt.plot([Rx + b[0]], [Ry + b[1]], marker='o', linewidth='5', color=color)

def main(theta_deg):
    theta = theta_deg * np.pi / 180
    colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00',
              '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']

    f = plt.figure()  # f = figure(n) if you know the figure number
    f.set_size_inches(11.69,8.27)
    rc('figure', figsize=(11.69,8.27))

    # MONOLAYER GRAPHENE UNROTATED
    a1=np.array([1/2, np.sqrt(3)/2])
    a2=np.array([-1/2, np.sqrt(3)/2])
    basis=[np.array([0, 0]), np.array([0, np.sqrt(3)/3])]

    # LAYER 1 ROTATED BY +THETA/2
    a11 = rotation(a1, theta/2)
    a12 = rotation(a2, theta/2)
    basis_1 = [rotation(basis[0], theta/2), rotation(basis[1], theta/2)]
    draw_2DLattice(a11, a12, basis_1, color=colors[0])

    # LAYER 2 ROTATED BY -THETA/2
    a21 = rotation(a1, -theta/2)
    a22 = rotation(a2, -theta/2)
    tt = -rotation(basis[1], -theta/2)      #### basis[1] is the vector connecting the two atoms in the unit cell
    basis_2 = [rotation(basis[0], -theta/2) + tt, rotation(basis[1], -theta/2) + tt]
    draw_2DLattice(a21, a22, basis_2, color=colors[1])

    #plt.xlabel(r'$x$', fontsize=20)
    #plt.ylabel(r'$y$', fontsize=20)
    plt.xlim(-4.6, 4.6)
    plt.ylim(-4.6, 4.6)
    #plt.legend(fontsize=12)
    #plt.grid(False)
    plt.title(r'rede TBG $\theta=%s^\circ$' % str(theta_deg))
    plt.gca().set_aspect('equal')
    plt.savefig("lattice.png", dpi=300, format='png', bbox_inches="tight")
    #plt.savefig("lattice.pdf", dpi=300, format='pdf')
    plt.clf()


def main2(theta_deg):
    theta = theta_deg * np.pi / 180
    colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00',
              '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']

    f = plt.figure()  # f = figure(n) if you know the figure number
    f.set_size_inches(11.69,8.27)
    rc('figure', figsize=(11.69,8.27))

    # MONOLAYER GRAPHENE UNROTATED
    a1=np.array([1/2, np.sqrt(3)/2])
    a2=np.array([-1/2, np.sqrt(3)/2])
    basis=[np.array([0, 0]), np.array([0, np.sqrt(3)/3])]

    # LAYER 1 UNROTATED
    a11 = rotation(a1, 0)
    a12 = rotation(a2, 0)
    basis_1 = [rotation(basis[0], 0), rotation(basis[1], 0)]
    draw_2DLattice(a11, a12, basis_1, color=colors[0])

    # LAYER 2 ROTATED BY THETA
    a21 = rotation(a1, theta)
    a22 = rotation(a2, theta)
    tt = -rotation(basis[1], theta)      #### basis[1] is the vector connecting the two atoms in the unit cell
    basis_2 = [rotation(basis[0], theta) + tt, rotation(basis[1], theta) + tt]
    draw_2DLattice(a21, a22, basis_2, color=colors[1])

    #plt.xlabel(r'$x$', fontsize=20)
    #plt.ylabel(r'$y$', fontsize=20)
    plt.xlim(-4.6, 4.6)
    plt.ylim(-4.6, 4.6)
    #plt.legend(fontsize=12)
    #plt.grid(False)
    plt.title(r'rede TBG $\theta=%s^\circ$' % str(theta_deg))
    plt.gca().set_aspect('equal')
    plt.savefig("lattice_%s.png" % str(theta_deg), dpi=300, format='png', bbox_inches="tight")
    #plt.savefig("lattice.pdf", dpi=300, format='pdf')
    plt.clf()

# unused code
#plt.savefig("plot.png", dpi=300, format='png', bbox_inches="tight")
#plt.clf()

if __name__ == '__main__':
    for theta_deg in [0, 3.89, 6.009, 9.43, 16.4264214, 21.786789, 27.795772496, 30,
                      32.2042275, 38.21321070173819, 42.10344887, 50.569992092103575, 60]:
        main(theta_deg)
