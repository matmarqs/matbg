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

def main():
    colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']
    f = plt.figure()  # f = figure(n) if you know the figure number
    f.set_size_inches(11.69,8.27)
    rc('figure', figsize=(11.69,8.27))

    a1 = np.array([1/2, np.sqrt(3)/2])
    a2 = np.array([-1/2, np.sqrt(3)/2])
    b = np.array([[   0,        0         ],     # crystal basis
                  [ 1/2, 1/(2*np.sqrt(3)) ]])
    delta = np.array([[  1/2, 1/(2*np.sqrt(3)) ],    # nearest neighbor vectors
                      [ -1/2, 1/(2*np.sqrt(3)) ],
                      [    0,    -1/np.sqrt(3) ]])
    # space group transformation
    R = np.array([[ np.cos(2*np.pi/6),np.sin(2*np.pi/6)],
                  [-np.sin(2*np.pi/6),np.cos(2*np.pi/6)]])
    tau = np.array([ 1/2, 1/(2*np.sqrt(3)) ])
    Sx = lambda R, tau, r: R[0][0] * r[0] + R[0][1] * r[1] + tau[0]
    Sy = lambda R, tau, r: R[1][0] * r[0] + R[1][1] * r[1] + tau[1]

    M = np.linspace(-5, 5, 11)
    N = np.linspace(-5, 5, 11)
    for x in M:
        for y in N:
            Vx = x * a1[0] + y * a2[0]
            Vy = x * a1[1] + y * a2[1]
            V = np.array([Vx, Vy])
            for i in range(len(delta)):
                plt.plot([Sx(R,tau,V), Sx(R,tau,V+delta[i])], [Sy(R,tau,V), Sy(R,tau,V+delta[i])], 'k-')
    for x in M:
        for y in N:
            Vx = x * a1[0] + y * a2[0]
            Vy = x * a1[1] + y * a2[1]
            V = np.array([Vx, Vy])
            for i in range(len(b)):
                plt.plot([Sx(R,tau,V+b[i])], [Sy(R,tau,V+b[i])], marker='o', markersize='10', color=colors[i])
    #plt.xlabel(r'$x$', fontsize=20)
    #plt.ylabel(r'$y$', fontsize=20)
    plt.xlim(-1.4, 1.4)
    plt.ylim(-1.4, 1.4)
    plt.axis('off')
    plt.grid(False)
    #plt.title(r'rede honeycomb')
    plt.gca().set_aspect('equal')
    plt.savefig("lattice.png", dpi=300, format='png', bbox_inches="tight")
    plt.clf()

# unused code
#plt.savefig("plot.png", dpi=300, format='png', bbox_inches="tight")
#plt.clf()

if __name__ == '__main__':
    main()
