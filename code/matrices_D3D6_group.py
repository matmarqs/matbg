#!/usr/bin/python3
from sympy import *

p = 2*pi/3

C6 = [Matrix([[cos(p/2), sin(p/2)], [-sin(p/2), cos(p/2)]]), "C6"]
Sv = [Matrix([[1, 0], [0, -1]]), "Sv"]
Sd = [Matrix([[-1, 0], [0, 1]]), "Sd"]

C65 = [C6[0]**5, "C65"]

C3 = [C6[0]**2, "C3"]
C32 = [C6[0]**4, "C32"]

C2 = [C6[0]**3, "C2"]

Svp = [Sv[0] * C3[0], "Sv'"]
Svpp = [Sv[0] * C32[0], "Sv''"]

Sdp = [Sd[0] * C3[0], "Sd'"]
Sdpp = [Sd[0] * C32[0], "Sd''"]

E1 = [Sv[0]**2, "E"]

group = [E1, C6, Sv, Sd, C65, C3, C32, C2, Svp, Svpp, Sdp, Sdpp]

def main_elem():
    g2 = Sv
    g = C3
    test_elem = [g[0] * g2[0], "C3 * Sv"]
    for elem in group:
        if test_elem[0] == elem[0]:
            print(test_elem[1], "=", elem[1])
    print(test_elem[0])

def main_diag():
    #A, D = C32[0].diagonalize()  # A D A^{-1}
    #print("A =", A)
    #print("D =", D)
    A = Matrix([[-I, I], [1, 1]])
    D = Matrix([[exp(2*pi*I/3), 0], [0, exp(4*pi*I/3)]])

    Result = A * Sv[0] * A**(-1)
    print(simplify(expand_complex(Result - Sv[0])))

main_diag()
