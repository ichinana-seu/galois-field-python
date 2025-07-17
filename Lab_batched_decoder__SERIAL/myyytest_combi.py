from GF2_map import GF2_map
import numpy as np
from itertools import combinations

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
    factorsetidx_1 = set(range(0, 12))
    factorsetidx_2 = set(range(0, 2))


    combinations1 = combinations(factorsetidx_1, 1)
    for combo1 in combinations1:
        combinations2 = combinations(factorsetidx_2, 1)
        for combo2 in combinations2:
            print("combinationsA: ")
            print(combo1)
            print("combinationsB: ")
            print(combo2)



    



    