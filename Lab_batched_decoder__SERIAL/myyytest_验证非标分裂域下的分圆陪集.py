from GF2_map import GF2_map
import numpy as np

# GF 2^2        Appendix        p(X) = 1 + X + X^2                  [1,1,1]
# GF 2^3        Appendix        p(X) = 1 + X^2 + X^3                [1,0,1,1]
# GF 2^4        Appendix        p(X) = 1 + X^3 + X^4                [1,0,0,1,1]
# GF 2^4        Book(ShuLin)    p(X) = 1 + X + X^4                  [1,1,0,0,1]
# GF 2^5        Appendix        p(X) = 1 + X^3 + X^5                [1,0,0,1,0,1]
# GF 2^6        Appendix        p(X) = 1 + X^5 + X^6                [1,0,0,0,0,1,1]
# GF 2^7        Appendix        p(X) = 1 + X^6 + X^7                [1,0,0,0,0,0,1,1]
# GF 2^8        Appendix        p(X) = 1 + X + X^6 + X^7 + X^8      [1,1,0,0,0,0,1,1,1]
# GF 2^9        Appendix        p(X) = 1 + X^5 + X^9                [1,0,0,0,0,1,0,0,0,1]
# GF 2^10       Appendix        p(X) = 1 + X^5 + X^10               [1,0,0,0,0,0,0,1,0,0,1]


if __name__ == "__main__":
    polynomial = np.array([1,1,0,0,0,0,1,1,1], dtype=np.int32)
    myGF2 = GF2_map(polynomial, 8)
    print(myGF2.table_exp2tuple)
    print(myGF2.table_tupleInt2exp)
    myGF2.print_elements_cyclotomicCoset()
    print(myGF2.order_of_element(3))
    myGF2.print_minimalPolynomials()

    myGF2.print_BCH_gx(t=3)



    