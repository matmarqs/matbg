#!/usr/bin/python3
from sympy import *

C3 =  Matrix([[-1, -1],
              [ 1,  0]])

C2 =  Matrix([[-1,  0],
              [ 0, -1]])

M11 = Matrix([[ 0,  1],
              [ 1,  0]])

C6 = C2 * C3**(-1)

E = C2**2

def print_orb(Orb):
    print(r"\qty{", end='');
    for q in Orb:
        print(r"\qty(\frac{%s}{%s},\frac{%s}{%s})" % (q[0].numerator,q[0].denominator,q[1].numerator,q[1].denominator), end=',')
    print(r"}", end='');

def main():
    q5 = Matrix([[Rational(1,4)], [Rational(0,1)]])
    q4 = Matrix([[Rational(1,6)], [Rational(1,6)]])
    q3 = Matrix([[Rational(1,2)], [Rational(0,1)]])
    q2 = Matrix([[Rational(1,3)], [Rational(1,3)]])
    q1 = Matrix([[Rational(0,1)], [Rational(0,1)]])

    Orb_q1 = [ E * q1 ]
    Orb_q2 = [ E * q2, C6 * q2 ]
    Orb_q3 = [ E * q3, C6 * q3, C6**2 * q3 ]
    Orb_q4 = [ E * q4, C6 * q4, C6**2 * q4, C6**3 * q4, C6**4 * q4, C6**5 * q4  ]
    Orb_q5 = [ E * q5, C6 * q5, C6**2 * q5, C6**3 * q5, C6**4 * q5, C6**5 * q5  ]

    print("Orb_q1: ")
    print_orb(Orb_q1)
    print()
    print("Orb_q2: ")
    print_orb(Orb_q2)
    print()
    print("Orb_q3: ")
    print_orb(Orb_q3)
    print()
    print("Orb_q4: ")
    print_orb(Orb_q4)
    print()
    print("Orb_q5: ")
    print_orb(Orb_q5)
    print()

if __name__ == '__main__':
    main()
