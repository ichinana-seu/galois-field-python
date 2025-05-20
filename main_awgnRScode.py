import numpy as np
import torch
import os
import math

from zhx_randomBinary_generator_Numpy import zhx_randomBinary_generator_Numpy
from GF2_map import GF2_map
import zhx_RS_EuclidDecoder


if __name__ == "__main__":
    gx = np.array( [ 6 , 9 , 6,  4, 14, 10,  0] , dtype=np.int32)
    RS_m = 4
    RS_t = 3
    RS_n = 2**RS_m-1


    primitive_polynomial = np.array([1,1,0,0,1], dtype=np.int32)
    myGF2map = GF2_map(primitive_polynomial, RS_m)
    myTestGen = zhx_randomBinary_generator_Numpy(RS_n, RS_m, RS_t,gx, myGF2map)
    

    Eval_ComplexSNR_dB_set = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for ComplexSNR_dB in Eval_ComplexSNR_dB_set: 
        error_blocks_now = 0
        BER_final = 1.0
        BLER_final = 1.0
        while error_blocks_now<50:
            # transmit
            ComplexSNR = 10 ** (ComplexSNR_dB / 10)
            noisepower = 1.0 / ComplexSNR
            noisepower_one_dim = noisepower / 2.0
            sigma_onedim = math.sqrt(noisepower_one_dim)
            codewords_poly = myTestGen.gen_codewords_poly()
            noise_bins = myTestGen.gen_awgn_noise(sigma_onedim)
            # channel
            received_numpy = myTestGen.Bpsk_transfer(codewords_poly, noise_bins)
            print(received_numpy)
            print(len(received_numpy))

            error_blocks_now += 1

            


