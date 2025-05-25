# version: 1 (2025-05-25)
# 适用于 GF(2^m) 的Galois扩域。请注意：这里的基域只能是2。
# 不可以是其他素数GF(p)->GF(p^m)或者GF(2^n)->GF(2^n^m)
# 表示法：幂次表示法

import numpy as np
import torch
from GF2_map import GF2_map

def zhx_RS_EuclidDecoder_inputPoly_torchBatch(received_poly_torch: torch.Tensor, RS_t: int, myGF2map: GF2_map, IOdevice: str):
    assert received_poly_torch.dim() == 2
    batchsize = received_poly_torch.shape[0]
    RS_m = myGF2map.m
    RS_n = 2**RS_m -1
    assert received_poly_torch.shape[1] == RS_n
    received_poly_numpy = received_poly_torch.cpu().numpy()
    corrected_polynomials = (-1) * np.ones( [batchsize, RS_n] , dtype=np.int32)
    for idx in range(0, batchsize):
        corrected_polynomials[idx, :], ___ = zhx_RS_EuclidDecoder_singlerow(received_poly_numpy[idx, :], RS_t, myGF2map)
    corrected_polynomials_torch = torch.from_numpy(corrected_polynomials).to(device=IOdevice, dtype=torch.int32)
    return corrected_polynomials_torch


def zhx_RS_EuclidDecoder_inputBins_torchBatch(received_bins_torch: torch.Tensor, RS_t: int, myGF2map: GF2_map, IOdevice: str):
    assert received_bins_torch.dim() == 2
    batchsize = received_bins_torch.shape[0]
    RS_m = myGF2map.m
    RS_n = 2**RS_m -1
    RS_binaryLength = RS_n*RS_m
    assert received_bins_torch.shape[1] == RS_binaryLength
    received_bins_numpy = received_bins_torch.cpu().numpy()
    corrected_bins_matrix = (-1) * np.ones( [batchsize, RS_binaryLength] , dtype=np.int32)
    for idx in range(0, batchsize):
        corrected_bins_matrix[idx, :], ___ = zhx_RS_EuclidDecoder_inputBins_singlerow(received_bins_numpy[idx, :], RS_t, myGF2map)
    corrected_bins_matrix_torch = torch.from_numpy(corrected_bins_matrix).to(device=IOdevice, dtype=torch.int32)
    return corrected_bins_matrix_torch

def zhx_RS_EuclidDecoder_singlerow(received_poly: np.ndarray, RS_t: int, myGF2map: GF2_map):
    RS_m = myGF2map.m
    RS_n = 2**RS_m -1
    assert len(received_poly) <= RS_n
    # 计算伴随式
    elements = np.arange(1, 2*RS_t+1, dtype=np.int32)
    poly_function_value_parallel = np.vectorize(myGF2map.poly_function_value, excluded=[0])
    syndrome = poly_function_value_parallel(received_poly, elements)   
    # 给output预留空间
    output = (-1) * np.ones(RS_n, dtype=np.int32)
    failflag = 1
   
    # 特殊情况，通过校验无需译码
    if np.all(syndrome == -1):
        output = received_poly.copy()
        failflag = 0
        return output, failflag

    # 正常情况
    quo_itera = (-1) * np.ones([2*RS_t+2, RS_n], dtype=np.int32)
    Z_function_itera = (-1) * np.ones([2*RS_t+2, RS_n], dtype=np.int32)
    sigma_function_itera = (-1) * np.ones([2*RS_t+2, RS_n], dtype=np.int32)
    ToBeDivided = (-1) * np.ones( [2*RS_t+1] , dtype=np.int32 )
    ToBeDivided[2*RS_t] = 0
    # itera_k = -1
    itera_k = -1
    Z_function_itera[itera_k+1, 0:len(ToBeDivided)] = ToBeDivided
    sigma_function_itera[itera_k+1, 0] = -1
    # itera_k = 0
    itera_k = 0
    Z_function_itera[itera_k+1, 0:len(syndrome)] = syndrome
    sigma_function_itera[itera_k+1, 0] = 0
    # itera_k = 1
    Divisor = myGF2map.poly_fresh(syndrome)
    itera_k = 1
    while True:
        q_tmp, z_tmp = myGF2map.poly_div_euclidmod(ToBeDivided, Divisor)
        quo_itera[itera_k+1, 0:len(q_tmp)] = q_tmp
        Z_function_itera[itera_k+1, 0:len(z_tmp)] = z_tmp
        sigma_func_tmp = myGF2map.poly_add( sigma_function_itera[itera_k+1 -2,:] , myGF2map.poly_addinverse( myGF2map.poly_mul( quo_itera[itera_k+1,:],   sigma_function_itera[itera_k+1 -1,:]      )        )      )
        sigma_function_itera[itera_k+1, 0:len(sigma_func_tmp)] = sigma_func_tmp
        # 中止条件
        condition1 = myGF2map.poly_degree(myGF2map.poly_fresh(Z_function_itera[itera_k+1, :])  )    <   myGF2map.poly_degree( myGF2map.poly_fresh(sigma_function_itera[itera_k+1, :])  )
        condition2 = myGF2map.poly_degree( myGF2map.poly_fresh(sigma_function_itera[itera_k+1, :])  )     <=    RS_t
        if condition1 and condition2:
            break
        # 准备下个循环
        ToBeDivided = Divisor
        Divisor = z_tmp
        if myGF2map.poly_degree(Divisor)==-1:       # 除数不可为0  # 异常处理
            output = received_poly.copy()
            failflag = 1
            return output, failflag
            raise ValueError('[ERROR] Divisor cannot be 0, you may catch the ValueError as failflag in RS decoding')
        itera_k = itera_k + 1

        # 超过迭代次数，也中止，直接报错即可
        if itera_k > 2*RS_t:
            failflag = 1
            output = received_poly
            return output, failflag
    # itera_k = 某个具体的itera_k，结束啦
    sigma_function_final = myGF2map.poly_fresh(sigma_function_itera[itera_k+1,:])
    Z_function_final = myGF2map.poly_fresh(Z_function_itera[itera_k+1,:])

    #print(Z_function_itera)
    #print(quo_itera)
    #print(sigma_function_itera)
    # try to find roots of sigma_function and take its mul_recipral
    errorlocation = []
    for ele in range(0,2**RS_m-1):
        if -1 == myGF2map.poly_function_value(sigma_function_final, ele):
            errorlocation.append( myGF2map.mulinverse(ele)  )
    errorlocation_index = np.array(errorlocation, dtype=np.int32)

    # RS码需要对sigma_function求导
    # 求得修正项
    sigma_function_final_derivative = myGF2map.poly_function_derivative(sigma_function_final)
    errorlocation_offset = []
    for i in range(0, len(errorlocation)):
        up = myGF2map.addinverse( myGF2map.poly_function_value(Z_function_final,   myGF2map.mulinverse( errorlocation[i] )   )      )
        down = myGF2map.poly_function_value(sigma_function_final_derivative,  myGF2map.mulinverse( errorlocation[i] ) )
        if down == -1:  # 除数不可为0  # 异常处理
            output = received_poly.copy()
            failflag = 1
            return output, failflag
            raise ValueError('[ERROR] Divisor cannot be 0, you may catch the ValueError as failflag in RS decoding')
        errorlocation_offset.append(myGF2map.mul(up, myGF2map.mulinverse(down) ) )
    errorlocation_offset_numpy = np.array(errorlocation_offset, dtype=np.int32)
    # 求得错误图样
    error_polynomial = (-1) * np.ones(RS_n, dtype=np.int32)
    error_polynomial[errorlocation_index] = errorlocation_offset_numpy
    corrected_polynomial = myGF2map.poly_add(received_poly, myGF2map.poly_addinverse(error_polynomial) )

    # valid answer?
    syndrome_new = poly_function_value_parallel(corrected_polynomial, elements)
    if np.all(syndrome_new == -1):
        failflag = 0
    # return
    return corrected_polynomial, failflag


def zhx_RS_EuclidDecoder_inputBins_singlerow(received_bins: np.ndarray, RS_t: int, myGF2map: GF2_map):
    RS_m = myGF2map.m
    RS_n = 2**RS_m -1
    RS_binaryLength = RS_n*RS_m
    assert len(received_bins) == RS_binaryLength
    received_poly = zhx_RS_binseq2poly(received_bins, myGF2map)
    # 计算伴随式
    elements = np.arange(1, 2*RS_t+1, dtype=np.int32)
    poly_function_value_parallel = np.vectorize(myGF2map.poly_function_value, excluded=[0])
    syndrome = poly_function_value_parallel(received_poly, elements)   
    # 给output预留空间
    output = (-1) * np.ones(RS_n, dtype=np.int32)
    failflag = 1
   
    # 特殊情况，通过校验无需译码
    if np.all(syndrome == -1):
        output = received_bins.copy()
        failflag = 0
        return output, failflag

    # 正常情况
    quo_itera = (-1) * np.ones([2*RS_t+2, RS_n], dtype=np.int32)
    Z_function_itera = (-1) * np.ones([2*RS_t+2, RS_n], dtype=np.int32)
    sigma_function_itera = (-1) * np.ones([2*RS_t+2, RS_n], dtype=np.int32)
    ToBeDivided = (-1) * np.ones( [2*RS_t+1] , dtype=np.int32 )
    ToBeDivided[2*RS_t] = 0
    # itera_k = -1
    itera_k = -1
    Z_function_itera[itera_k+1, 0:len(ToBeDivided)] = ToBeDivided
    sigma_function_itera[itera_k+1, 0] = -1
    # itera_k = 0
    itera_k = 0
    Z_function_itera[itera_k+1, 0:len(syndrome)] = syndrome
    sigma_function_itera[itera_k+1, 0] = 0
    # itera_k = 1
    Divisor = myGF2map.poly_fresh(syndrome)
    itera_k = 1
    while True:
        q_tmp, z_tmp = myGF2map.poly_div_euclidmod(ToBeDivided, Divisor)
        quo_itera[itera_k+1, 0:len(q_tmp)] = q_tmp
        Z_function_itera[itera_k+1, 0:len(z_tmp)] = z_tmp
        sigma_func_tmp = myGF2map.poly_add( sigma_function_itera[itera_k+1 -2,:] , myGF2map.poly_addinverse( myGF2map.poly_mul( quo_itera[itera_k+1,:],   sigma_function_itera[itera_k+1 -1,:]      )        )      )
        sigma_function_itera[itera_k+1, 0:len(sigma_func_tmp)] = sigma_func_tmp
        # 中止条件
        condition1 = myGF2map.poly_degree(myGF2map.poly_fresh(Z_function_itera[itera_k+1, :])  )    <   myGF2map.poly_degree( myGF2map.poly_fresh(sigma_function_itera[itera_k+1, :])  )
        condition2 = myGF2map.poly_degree( myGF2map.poly_fresh(sigma_function_itera[itera_k+1, :])  )     <=    RS_t
        '''
        print(f"itera = {itera_k}, condition1 = {condition1}, condition2 = {condition2}")
        print(f"z {Z_function_itera[itera_k+1, :]}")
        print(f"quo_itera {quo_itera[itera_k+1, :]}")
        print(f"sigma {sigma_function_itera[itera_k+1, :]}")
        '''
        if condition1 and condition2:
            break
        # 准备下个循环
        ToBeDivided = Divisor
        Divisor = z_tmp
        if myGF2map.poly_degree(Divisor)==-1:       # 除数不可为0  # 异常处理
            output = received_bins.copy()
            failflag = 1
            return output, failflag
            raise ValueError('[ERROR] Divisor cannot be 0, you may catch the ValueError as failflag in RS decoding')
        itera_k = itera_k + 1

        # 超过迭代次数，也中止，直接报错即可
        if itera_k > 2*RS_t:
            failflag = 1
            output = received_bins.copy()
            return output, failflag
    # itera_k = 某个具体的itera_k，结束啦
    sigma_function_final = myGF2map.poly_fresh(sigma_function_itera[itera_k+1,:])
    Z_function_final = myGF2map.poly_fresh(Z_function_itera[itera_k+1,:])

    #print(Z_function_itera)
    #print(quo_itera)
    #print(sigma_function_itera)
    # try to find roots of sigma_function and take its mul_recipral
    errorlocation = []
    for ele in range(0,2**RS_m-1):
        if -1 == myGF2map.poly_function_value(sigma_function_final, ele):
            errorlocation.append( myGF2map.mulinverse(ele)  )
    errorlocation_index = np.array(errorlocation, dtype=np.int32)

    # RS码需要对sigma_function求导
    # 求得修正项
    sigma_function_final_derivative = myGF2map.poly_function_derivative(sigma_function_final)
    errorlocation_offset = []
    for i in range(0, len(errorlocation)):
        up = myGF2map.addinverse( myGF2map.poly_function_value(Z_function_final,   myGF2map.mulinverse( errorlocation[i] )   )      )
        down = myGF2map.poly_function_value(sigma_function_final_derivative,  myGF2map.mulinverse( errorlocation[i] ) )
        if down == -1:  # 除数不可为0  # 异常处理
            output = received_bins.copy()
            failflag = 1
            return output, failflag
            raise ValueError('[ERROR] Divisor cannot be 0, you may catch the ValueError as failflag in RS decoding')
        errorlocation_offset.append(myGF2map.mul(up, myGF2map.mulinverse(down) ) )
    errorlocation_offset_numpy = np.array(errorlocation_offset, dtype=np.int32)
    # 求得错误图样
    error_polynomial = (-1) * np.ones(RS_n, dtype=np.int32)
    error_polynomial[errorlocation_index] = errorlocation_offset_numpy
    corrected_polynomial = myGF2map.poly_add(received_poly, myGF2map.poly_addinverse(error_polynomial) )

    # valid answer?
    syndrome_new = poly_function_value_parallel(corrected_polynomial, elements)
    if np.all(syndrome_new == -1):
        failflag = 0
    # return
    corrected_bins = zhx_RS_poly2binseq(corrected_polynomial, myGF2map)
    return corrected_bins, failflag


def zhx_RS_poly2binseq(poly: np.ndarray, myGF2map: GF2_map):
    m = myGF2map.m
    n = 2**m -1
    assert len(poly) <= n
    if np.any(poly<-1 )or np.any(poly>2**m-2):
        raise ValueError(f"[ERROR] Power must be >= -1 and <=2**m-2 .")
    binseq = np.zeros( m*n ,dtype=np.int32)
    for index in range(0, len(poly) ):
        binseq[index*m: (index+1)*m] = myGF2map.convert_exp2tuple(poly[index])
    return binseq

def zhx_RS_binseq2poly(binseq: np.ndarray, myGF2map: GF2_map):
    m = myGF2map.m
    n = 2**m -1
    assert len(binseq) == m*n
    if len(binseq)%m != 0:
        raise ValueError(f"[ERROR] Input binary sequence, length must be a multiple of {m} .")
    binseq_grouped = binseq.reshape([-1, m])
    assert binseq_grouped.ndim == 2
    power = (-99) * np.ones( [binseq_grouped.shape[0]] ,dtype=np.int32)
    for row in range(0, binseq_grouped.shape[0] ):
        power[row] = myGF2map.convert_tuple2exp(binseq_grouped[row,:])
    return power













########################### 以下为测试 ##########################
if __name__ == "__main__":

    '''
    # 课本上的例子
    primitive_polynomial = np.array([1,1,0,0,1], dtype=np.int32)
    myGF2 = GF2_map(primitive_polynomial, 4)

    received_poly = np.array([-1,  -1, -1,  7,  -1,  -1,  -1,  -1,  -1,  -1,  11,  -1,  -1,  -1,  -1], dtype=np.int32)
    RS_t = 3
    corrected_polynomial, failflag = zhx_RS_EuclidDecoder(received_poly, RS_t, myGF2)
    print(f"failflag = {failflag}")
    print(f"corrected_polynomial = {corrected_polynomial}")
    corrected_bins = zhx_RS_poly2binseq(corrected_polynomial, myGF2)
    print(f"corrected_bins = {corrected_bins}")
    '''





    # 一个报错的例子
    primitive_polynomial = np.array([1,1,0,0,1], dtype=np.int32)
    myGF2 = GF2_map(primitive_polynomial, 4)

    # received_poly = np.array([ -1, -1, -1, 7, -1, -1, 9, -1, -1, -1, 11, -1, -1, -1, -1 ], dtype=np.int32)
    received_poly = np.array([-1,  6, 10,  6,  4,  7,  2,  1,  8,  2,  6,  3,  3,  0,  3], dtype=np.int32)
    received_bins = zhx_RS_poly2binseq(received_poly, myGF2)
    print(f"received_poly = {received_poly}")
    print(f"received_bins = {received_bins}")

    RS_t = 3
    corrected_polynomial, failflag = zhx_RS_EuclidDecoder_singlerow(received_poly, RS_t, myGF2)
    print(f"failflag = {failflag}")
    print(f"corrected_polynomial = {corrected_polynomial}")
    corrected_bins = zhx_RS_poly2binseq(corrected_polynomial, myGF2)
    print(f"corrected_bins = {corrected_bins}")

