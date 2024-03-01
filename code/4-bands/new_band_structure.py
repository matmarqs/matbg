#!/usr/bin/env python3

import numpy as np

from matplotlib import pyplot as plt
## comentar as 4 linhas abaixo caso nao tenha o LaTeX no matplotlib ###
from matplotlib import rc
plt.style.use('bmh')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{physics} \usepackage{bm}')
#matplotlib.verbose.level = 'debug-annoying'
#######################################################################
#from scipy.integrate import simpson

N_line=100

class k_point:
    def __init__(self, label, kx_ky):
        self.label = label  # LaTeX label. Example: r'$\Gamma$' for the Gamma point
        self.k = kx_ky      # [kx, ky] numpy array

def dist(p1, p2):
    return np.sqrt(sum((p1.k - p2.k)**2))

# give the energy eigenvalues (bands) associated with the momentum point k
def eigenvals(k):
    return [np.cos(k[0]) + np.cos(k[1]), 2*np.cos(k[0]) + 2*np.cos(k[1])]

# path is a list of momentum points with their labels. Example: 'Gamma', 'X', 'L', 'Gamma'
# eigvals is a function that return the energy eigenvalues associated with momentum k
# n_line is the number of k points for each line. Example: 100 points for the 'Gamma' -> 'X' line
def gen_band(path, eigvals, n_line=100):
    bands = []
    for index in range(len(path)-1):     # Example: index refers to line 1 'Gamma' -> 'X', line 2 'X' -> 'L', line 3 'L' -> 'Gamma'
        for i in range(n_line):
            z = i / n_line
            k = path[index].k * (1-z) + z * path[index+1].k
            bands.append(eigvals(k))
    bands = np.array(bands)
    return bands    # bands is a M x N matrix, where M = ((len(path) - 1) * n_line) and N = len(eigvals)

def ticks_positions(path):

def plot_band(path, bands):
    b = np.transpose(bands)
    n_kpoints = len(energy)
    plt.plot(range(n_kpoints), energy)

def main():
    labels = [r'$\Gamma$', r'$X$', r'$L$', r'$\Gamma$']
    k_points = np.array([[0, 0],       # Gamma
                         [1, 0],       # X
                         [1, 1],       # L
                         [0, 0]])      # Gamma
    path = []
    for i in range(len(labels)):
        path.append(k_point(labels[i], k_points[i]))

    bands = gen_band(path, eigenvals, n_line=N_line)
    plot_band(path_1, energy)

    ticks_pos = [0] # position of the x_ticks at the 0 to 1 scale
    s_dist = 0      # cumulative sum of distances at each step
    total_dist = sum([dist(path[i], path[i+1]) for i in range(len(path)-1)])    # total path distance
    for i in range(len(path)-1):
        s_dist += dist(path[i], path[i+1]) / total_dist
        ticks_pos.append(s_dist)

    #plt.axhline(y=0, ls='--', color='k')
    plt.xlim(ticks_pos[0], ticks_pos[-1])
    plt.xticks(ticks_pos, labels, fontsize=20)
    plt.grid(True, axis='x')
    #######ax.set_xticks([0.15, 0.68, 0.97])
    #######ax.set_yticks([0.2, 0.55, 0.76])

    ymin = 0; ymax = 5.10
    plt.ylim(ymin, ymax);
    plt.ylabel(r'energy', fontsize=20)
    plt.savefig("band_struct_test.png", dpi=300, format='png', bbox_inches="tight")

if __name__ == '__main__':
    main()
