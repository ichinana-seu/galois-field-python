# version: 3 (2025-05-19)
# 适用于 GF(2^m) 的Galois扩域。请注意：这里的基域只能是2。
# 不可以是其他素数GF(p)->GF(p^m)或者GF(2^n)->GF(2^n^m)
# 表示法：幂次表示法

import numpy as np
from GF2_map import GF2_map


# 输入的应该是长度为2^m-1的 “-1或0”，输出也一样。如果要将幂级数表示法转换为010101二进制，则 【二进制=幂级数+1】 即可
def zhx_BCH_BerlekampMasseyDecoder(received_polynomial: np.ndarray, BCH_t: int, myGF2map: GF2_map):
    BCH_m = myGF2map.m
    BCH_n = 2**BCH_m -1
    assert len(received_polynomial) <= BCH_n
    # 计算伴随式
    elements = np.arange(1, 2*BCH_t+1, dtype=np.int32)
    poly_function_value_parallel = np.vectorize(myGF2map.poly_function_value, excluded=[0])
    syndrome = poly_function_value_parallel(received_polynomial, elements)     
    # 给output预留空间
    output = (-1) * np.ones(BCH_n, dtype=np.int32)
    failflag = 1

    # 特殊情况，通过校验无需译码
    if np.all(syndrome == -1):
        output = received_polynomial
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
    discrepancy_itera[1] = syndrome[0]                                                                                                  # Ex: alpha^0
    sigma_function_itera[1, 0] = 0
    l_degreeOfSigmaFunc_itera[1] = 0
    LminusK_stepdiff_itera[1] = itera_k - l_degreeOfSigmaFunc_itera[1]          #   = 0
    # 预先计算下一个迭代的sigma_function
    max_stepdiff_indexi = -1
    correction_term = myGF2map.mul( discrepancy_itera[0+1] , myGF2map.mulinverse( discrepancy_itera[max_stepdiff_indexi+1]   ) )        # Ex: = symd0 = alpha^0
    tmp = (-1) * np.ones( [itera_k-max_stepdiff_indexi+1] , dtype=np.int32 )
    tmp[itera_k-max_stepdiff_indexi] = 0                
    correction_term = myGF2map.poly_mul( np.array([correction_term], dtype=np.int32) ,  tmp )                                                  # Ex: = [-1, symd0]            
    correction_term = myGF2map.poly_mul( correction_term, myGF2map.poly_fresh( sigma_function_itera[max_stepdiff_indexi+1, :] )  )           # Ex: = [-1, symd0] 
    next_sigma_function = myGF2map.poly_fresh( myGF2map.poly_add( sigma_function_itera[itera_k+1, :] ,   correction_term  ) )
    sigma_function_itera[itera_k+2, 0:len(next_sigma_function)] = next_sigma_function                                                   # Ex: = [0, symd0]

    # itera_k = 1 ... 2t-1
    for itera_k in range(1, 2*BCH_t):
        l_degreeOfSigmaFunc_itera[itera_k+1] = len(myGF2map.poly_fresh(sigma_function_itera[itera_k+1, :])) - 1
        '''print(f"l_degreeOfSigmaFunc_itera = {l_degreeOfSigmaFunc_itera[itera_k+1]}")     '''
        LminusK_stepdiff_itera[itera_k+1] = itera_k - l_degreeOfSigmaFunc_itera[itera_k+1]
        '''print(f"LminusK_stepdiff_itera = {LminusK_stepdiff_itera[itera_k+1]}")       '''
        # 计算d_k，为下一次迭代做准备
        d_tmp = syndrome[itera_k]
        for lower_index in range(1, l_degreeOfSigmaFunc_itera[itera_k+1]+1):
            d_tmp = myGF2map.add(d_tmp,       myGF2map.mul( sigma_function_itera[itera_k+1, lower_index] , syndrome[itera_k-lower_index] )    )
        discrepancy_itera[itera_k+1] = d_tmp
        '''print(f"discrepancy_itera = {discrepancy_itera[itera_k+1]}")     '''
        # 预先计算下一个迭代的sigma_function
        if d_tmp == -1: 
            sigma_function_itera[itera_k+2, :] = sigma_function_itera[itera_k+1, :]
        else:       # 搜寻第 -1 ～ k-1 步，选择第i步，满足d_i ~= alpha^(-1)，且 「步异」i−l_i最大
            candidate_elements = LminusK_stepdiff_itera[0:itera_k+1]
            mask = ( candidate_elements != -1)
            if np.any(mask):
                dedicate_elements = candidate_elements[mask]
                max_valid = np.max(dedicate_elements)
                max_index = np.where(candidate_elements == max_valid)[0][0].item()
                assert type(max_index) == int
            else:
                raise NotImplementedError("[ERROR] Cannot find max stepdiff index i which also !=alpha^(-1)")
            max_stepdiff_indexi = max_index-1
            '''print(f"max_stepdiff_indexi = {max_stepdiff_indexi}")        '''
            # 修正项
            correction_term = myGF2map.mul( discrepancy_itera[itera_k+1] , myGF2map.mulinverse( discrepancy_itera[max_stepdiff_indexi+1]   ) )        # Ex: = symd0 = alpha^0
            tmp = (-1) * np.ones( [itera_k-max_stepdiff_indexi+1] , dtype=np.int32 )
            tmp[itera_k-max_stepdiff_indexi] = 0                                                                                                # Ex: [-1, 0]   = x^1
            correction_term = myGF2map.poly_mul( np.array([correction_term], dtype=np.int32) ,  tmp )                                                  # Ex: = [-1, symd0]            
            correction_term = myGF2map.poly_mul( correction_term, myGF2map.poly_fresh( sigma_function_itera[max_stepdiff_indexi+1, :] )  )           # Ex: = [-1, symd0] 
            next_sigma_function = myGF2map.poly_fresh( myGF2map.poly_add( sigma_function_itera[itera_k+1, :] ,   correction_term  ) )
            sigma_function_itera[itera_k+2, 0:len(next_sigma_function)] = next_sigma_function        
        '''print(f"\n\nitera_k = {itera_k + 1}")        '''
        '''print(f"sigma_function_itera = {sigma_function_itera[itera_k+2, 0:len(next_sigma_function)]}")         '''               
    # itera_k = 2t 结束啦
    sigma_function_final = myGF2map.poly_fresh(sigma_function_itera[2*BCH_t+1,:])

    # try to find roots of sigma_function and take its mul_recipral
    errorlocation = []
    for ele in range(0,2**BCH_m-1):
        if -1 == myGF2map.poly_function_value(sigma_function_final, ele):
            errorlocation.append( myGF2map.mulinverse(ele)  )
    errorlocation_index = np.array(errorlocation, dtype=np.int32)
    error_polynomial = (-1) * np.ones(BCH_n, dtype=np.int32)
    error_polynomial[errorlocation_index] = 0
    corrected_polynomial = myGF2map.poly_add(received_polynomial, myGF2map.poly_addinverse(error_polynomial) )

    # valid answer?
    syndrome_new = poly_function_value_parallel(corrected_polynomial, elements)
    if np.all(syndrome_new == -1):
        failflag = 0
    # return
    return corrected_polynomial, failflag

            
def zhx_BCH_poly2bin(poly: np.ndarray, lenth: int):
    bin = (-1) * np.ones([lenth], dtype=np.int32)
    bin[0: len(poly)] = poly
    bin = bin + 1
    return bin

def zhx_BCH_bin2poly(bin: np.ndarray):
    results = bin - 1
    while(1):
        if results[len(results)-1]==-1 and len(results)!=1:
            results = results[0: len(results)-1]
        else:
            break
    return results













########################### 以下为测试 ##########################
if __name__ == "__main__":
    primitive_polynomial = np.array([1,1,0,0,1], dtype=np.int32)
    myGF2 = GF2_map(primitive_polynomial, 4)

    gx_bin = np.array([1, 1, 1, 0, 1, 1 ,0 ,0 ,1 ,0 ,1], dtype=np.int32)
    gx = gx_bin - 1
    received_bin = np.array([0, 0, 0 ,1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0 ], dtype=np.int32)
    received = zhx_BCH_bin2poly(received_bin)
    print(f"received_bin = {received_bin}")

    BCH_t = 3
    corrected_polynomial, failflag = zhx_BCH_BerlekampMasseyDecoder(received, BCH_t, myGF2)
    corrected_polynomial_bin = zhx_BCH_poly2bin(corrected_polynomial, 2**4-1)
    print(f"failflag = {failflag}")
    print(f"corrected_polynomial_bin = {corrected_polynomial_bin}")
