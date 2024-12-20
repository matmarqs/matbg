#!/usr/bin/python3
from sympy import *

C6 =  Matrix([[cos(pi/6) + I * sin(pi/6), 0],
              [ 0,  cos(pi/6) - I * sin(pi/6)]])

M1c1 = Matrix([[ 0,  I],
              [ I,  0]])

C3 = simplify(expand_complex(C6**2))
C2 = simplify(expand_complex(C6**3))

M1c1p = M1c1 * C3
M1c1pp = M1c1 * C3**2

Sd = C2 * M1c1
Sdp = Sd * C3
Sdpp = Sd * C3**2

Ebar = simplify(expand_complex(C2**2))

E = Ebar**2

group = [E, C6, C3, C2, C3**2, C6**5, Sd, Sdp, Sdpp, M1c1, M1c1p, M1c1pp]
double_group = group + [Ebar * g for g in group]
G = [simplify(expand_complex(g)) for g in double_group]
G_labels = [r"E", r"C_6", r"C_3", r"C_2", r"C_3^2", r"C_6^5", r"\sigma_d", r"\sigma_d'", r"\sigma_d''", r"m_{1\cc{1}}", r"m_{1\cc{1}}'", r"m_{1\cc{1}}''",
            r"\cc{E}", r"\cc{C}_6", r"\cc{C}_3", r"\cc{C}_2", r"\cc{C}_3^2", r"\cc{C}_6^5", r"\cc{\sigma}_d", r"\cc{\sigma}_d'", r"\cc{\sigma}_d''", r"\cc{m}_{1\cc{1}}", r"\cc{m}_{1\cc{1}}'", r"\cc{m}_{1\cc{1}}''"]

def print_orb(Orb):
    print(r"\qty{", end='');
    for q in Orb:
        print(r"\qty(\frac{%s}{%s},\frac{%s}{%s})" % (q[0].numerator,q[0].denominator,q[1].numerator,q[1].denominator), end=',')
    print(r"}", end='');

def return_index(L, elem):
    for i in range(len(L)):
        if L[i] == elem:
            return i
    return -1

def main():
    #myelem = C6**(-1) * M1c1 * C6
    myelem = M1c1 * C3

    MyElem = simplify(expand_complex(myelem))
    print(return_index(G, MyElem))
    print(G_labels[return_index(G, MyElem)])

    #for g in G:
    #    print(g)

if __name__ == '__main__':
    main()
