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

Svp = [C65[0] * Sv[0] * C6[0], "Sv'"]
Svpp = [C32[0] * Sv[0] * C3[0], "Sv''"]

Sdp = [C65[0] * Sd[0] * C6[0], "Sd'"]
Sdpp = [C32[0] * Sd[0] * C3[0], "Sd''"]

E1 = [Sv[0]**2, "E"]

group = [E1, C6, Sv, Sd, C65, C3, C32, C2, Svp, Svpp, Sdp, Sdpp]

def main():
    #g = C6
    #g2 = Sd

    #test_elem = [Sv[0] * Sd[0], "Sv * Sd"]
    test_elem = Sv

    for elem in group:
        if test_elem[0] == elem[0]:
            print(test_elem[1], "=", elem[1])

    print(test_elem[0])

main()
