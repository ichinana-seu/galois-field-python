import numpy as np
from GF2_map import GF2_map

def zhx_BCH_BerlekampMasseyDecoder(received: np.ndarray, BCH_t: int, myGF2map: GF2_map):
    BCH_m = myGF2map.m
    BCH_n = 2**BCH_m -1
    # 计算伴随式
    elements = np.arange(1, 2*BCH_t+1, dtype=np.int32)
    poly_function_value_parallel = np.vectorize(myGF2map.poly_function_value, excluded=[0])
    syndrome = poly_function_value_parallel(received, elements)     
    # 给output预留空间
    output = (-1) * np.ones(BCH_n, dtype=np.int32)
    failflag = 1


    # 特殊情况，通过校验无需译码
    if np.all(syndrome == 0):
        output = received
        failflag = 0
        return output, failflag

    # 正常情况
    discrepancy_itera = (-99) * np.ones(2*BCH_t+1, dtype=np.int32)
    l_degreeOfSigmaFunc_itera = np.zeros(2*BCH_t+1, dtype=np.int32)
    LminusK_stepdiff_itera = (-99) * np.ones(2*BCH_t+1, dtype=np.int32)
    sigma_function_itera = (-1) * np.ones([2*BCH_t+2, BCH_n], dtype=np.int32)
    # itera_k = -1
    discrepancy_itera[0] = 0
    sigma_function_itera[0, 0] = 0
    l_degreeOfSigmaFunc_itera[0] = 0
    LminusK_stepdiff_itera[0] = -1
    # itera_k = 0
    itera_k = 0
    discrepancy_itera[1] = syndrome[0]
    sigma_function_itera[1, 0] = 0
    l_degreeOfSigmaFunc_itera[1] = 0
    LminusK_stepdiff_itera[1] = itera_k - l_degreeOfSigmaFunc_itera[1]          #   = 0
    # 预先计算下一个迭代的sigma_function
    max_stepdiff_indexi = -1
    correction_term = myGF2map.mul( discrepancy_itera[0+1] , myGF2map.mulinverse( discrepancy_itera[max_stepdiff_indexi+1]   ) )
    tmp = (-1) * np.ones( [0-max_stepdiff_indexi+1] , dtype=np.int32 )
    tmp[0-max_stepdiff_indexi]
    correction_term = myGF2map.mul( np.array(correction_term, dtype=np.int32) ,  tmp )
    correction_term = myGF2map.mul( correction_term, myGF2map.poly_fresh( sigma_function_itera[max_stepdiff_indexi+1, :] )  )

    # itera_k = 1
    for itera_k in range(1, 2*BCH_t):
        # select the most giant setdiff
        max_stepdiff_indexi = np.argmax( LminusK_stepdiff_itera[0:]  ) - 1


        previous_sigma_function
        previous_sigma_function = myGF2map.poly_fresh( sigma_function_itera[itera_k, :] )
        correction_term = 


        sigma_function_itera[]
        l_degreeOfSigmaFunc_itera[itera_k+1] = len(myGF2map.poly_fresh(sigma_function_itera)) - 1

        # 计算d_k，为下一次迭代做准备
        discrepancy_itera[itera_k+1] = syndrome[1] + sigma_function_itera[]

        # 预先计算下一个迭代的sigma_function



    # last: itera_k = 2t
    sigma_function = sigma_function_itera[2*BCH_t+1]
    pass

if __name__ == "__main__":
    primitive_polynomial = np.array([1,1,0,0,1], dtype=np.int32)
    myGF2 = GF2_map(primitive_polynomial, 4)

    gx_bin = np.array([1, 1, 1, 0, 1, 1 ,0 ,0 ,1 ,0 ,1], dtype=np.int32)
    gx = gx_bin - 1
    received_bin = np.array([0, 0, 0 ,1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0 ], dtype=np.int32)
    received = received_bin - 1

    

    BCH_t = 3
    zhx_BCH_BerlekampMasseyDecoder(received, BCH_t, myGF2)