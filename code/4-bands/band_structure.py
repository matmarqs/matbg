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

# give the energy eigenvalues (bands) associated with the momentum point k
def eigenval_func_graphene(k):
    a = 2.46   # monolayer graphene lattice constant (in Angstrons)
    a1 = a * np.array([ 1/2, np.sqrt(3)/2])  # unrotated 1st lattice vector a1
    a2 = a * np.array([-1/2, np.sqrt(3)/2])  # unrotated 2nd lattice vector a2
    # nearest neighbor vectors
    delta1 = ((a1 + a2)/3)
    delta2 = ((-2*a1 + a2)/3)
    delta3 = ((a1 - 2*a2)/3)
    f = lambda k: np.exp(1j*np.dot(k, delta1)) + np.exp(1j*np.dot(k, delta2)) + np.exp(1j*np.dot(k, delta3))
    t = 1.0 # monolayer hopping t = 2.8 eV. See Castro Neto "https://doi.org/10.1103/RevModPhys.81.109"
    return [-t * np.abs(f(k)), t * np.abs(f(k))]

# k point with its label and its coordinates
class k_point:
    def __init__(self, label, kx_ky):
        self.label = label            # LaTeX label. Example: r'$\Gamma$' for the Gamma point
        self.k = np.array(kx_ky)      # [kx, ky] numpy array

# euclidian distance between two k points
def dist(p1, p2):
    return np.sqrt(sum((p1.k - p2.k)**2))

# path is a list of momentum points with their labels. Example: 'Gamma', 'X', 'L', 'Gamma'
# eigvals is a function that return the energy eigenvalues associated with momentum k
# n_line is the number of k points for each line. Example: 100 points for the 'Gamma' -> 'X' line
def gen_band(path, eigvals_func, n_line=100):
    bands = []
    for index in range(len(path)-1):     # Example: index refers to line 1 'Gamma' -> 'X', line 2 'X' -> 'L', line 3 'L' -> 'Gamma'
        for i in range(n_line):
            z = i / n_line
            k = path[index].k * (1-z) + z * path[index+1].k
            bands.append(eigvals_func(k))
    bands = np.array(bands)
    return bands    # bands is a M x N matrix, where M = ((len(path) - 1) * n_line) and N = len(eigvals)

# return the array of x_ticks positions on a 0 to 1 scale
def ticks_positions(path):
    ticks_pos = [0] # position of the x_ticks at the 0 to 1 scale
    s_dist = 0      # cumulative sum of distances at each step
    total_dist = sum([dist(path[i], path[i+1]) for i in range(len(path)-1)])    # total path distance
    for i in range(len(path)-1):
        s_dist += dist(path[i], path[i+1]) / total_dist
        ticks_pos.append(s_dist)
    return ticks_pos

# plot bands
def plot_bands(bands, ticks_pos, n_line=100):
    x = np.concatenate([np.linspace(ticks_pos[i], ticks_pos[i+1], n_line, endpoint=False) for i in range(len(ticks_pos)-1)], axis=None)
    bands_t = np.transpose(bands)
    for b in bands_t:
        plt.plot(x, b)

# plot the bandstructure along path
def plot_bandstructure(path, eigenval_func, ticks_fontsize=20, n_line=100):
    labels = [ kp.label for kp in path ]
    ticks_pos = ticks_positions(path)
    bands = gen_band(path, eigenval_func, n_line=n_line)
    plot_bands(bands, ticks_pos, n_line=n_line)
    #plt.axhline(y=0, ls='--', color='k')
    plt.xlim(ticks_pos[0], ticks_pos[-1])
    plt.xticks(ticks_pos, labels, fontsize=20)
    plt.grid(True, axis='x')

def test_1():
    a = 2.46   # monolayer graphene lattice constant (in Angstrons)
    path = [
             k_point(r'$\Gamma$',   [0, 0]),                            # Gamma
             k_point(r'$K$', [4*np.pi/(3*a), 0]),                       # K
             k_point(r'$M$',        [np.pi/a, np.pi/(np.sqrt(3)*a)]),   # M
             k_point(r'$\Gamma$',   [0, 0]),                            # Gamma
           ]
    plot_bandstructure(path, eigenval_func=eigenval_func_graphene, ticks_fontsize=20, n_line=100)
    #ymin = 0; ymax = 4.10
    #plt.ylim(ymin, ymax);
    plt.ylabel(r'energy', fontsize=20)
    plt.savefig("band_struct_test-graphene1.png", dpi=300, format='png', bbox_inches="tight")
    plt.clf()

def test_2():
    a = 2.46   # monolayer graphene lattice constant (in Angstrons)
    path = [
             k_point(r'$K$',        [4*np.pi/(3*a), 0]),                        # K
             k_point(r"$K'$",       [2*np.pi/(3*a), 2*np.pi/(np.sqrt(3)*a)]),   # K'
             k_point(r'$\Gamma$',   [0, 0]),                                    # Gamma
             k_point(r'$M$',        [np.pi/a, np.pi/(np.sqrt(3)*a)]),           # M
             k_point(r'$K$',        [4*np.pi/(3*a), 0]),                        # K
           ]
    plot_bandstructure(path, eigenval_func=eigenval_func_graphene, ticks_fontsize=20, n_line=100)
    plt.ylabel(r'energy', fontsize=20)
    plt.savefig("band_struct_test-graphene2.png", dpi=300, format='png', bbox_inches="tight")
    plt.clf()

if __name__ == '__main__':
    test_1()
    test_2()
