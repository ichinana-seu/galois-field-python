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
    RS_binaryLength = RS_n*RS_m


    primitive_polynomial = np.array([1,1,0,0,1], dtype=np.int32)
    myGF2map = GF2_map(primitive_polynomial, RS_m)
    myTestGen = zhx_randomBinary_generator_Numpy(RS_n, RS_m, RS_t,gx, myGF2map)
    

    Eval_ComplexSNR_dB_set = [-1, 0, 1, 2, 3, 4]
    for ComplexSNR_dB in Eval_ComplexSNR_dB_set: 
        batch_id = 0
        error_blocks_now = 0
        BER_final = 1.0
        BLER_final = 1.0
        while error_blocks_now < 50:
            # transmit
            ComplexSNR = 10 ** (ComplexSNR_dB / 10)
            noisepower = 1.0 / ComplexSNR
            noisepower_one_dim = noisepower / 2.0
            sigma_onedim = math.sqrt(noisepower_one_dim)
            codewords_poly = myTestGen.gen_codewords_poly()
            codewords_bins = zhx_RS_EuclidDecoder.zhx_RS_poly2binseq(codewords_poly, myGF2map)
            noise_bins = myTestGen.gen_awgn_noise(sigma_onedim)
            # channel
            received_bins = myTestGen.Bpsk_transfer(codewords_poly, noise_bins)
            # pre-process
            received_hardDecision = (received_bins<0).astype(np.int32)
            received_hardDecision_poly = zhx_RS_EuclidDecoder.zhx_RS_binseq2poly(received_hardDecision, myGF2map)
            hardout_poly, failflag = zhx_RS_EuclidDecoder.zhx_RS_EuclidDecoder(received_hardDecision_poly, RS_t, myGF2map)
            hardout_bins = zhx_RS_EuclidDecoder.zhx_RS_poly2binseq(hardout_poly, myGF2map)
            # 统计BER
            diff_bits = (hardout_bins != codewords_bins)
            diff_bits_num = np.count_nonzero(diff_bits)
            BER_thisbatch = diff_bits_num / RS_binaryLength
            BER_final = batch_id / (batch_id + 1) * BER_final + 1 / (batch_id + 1) * BER_thisbatch
            # 统计BLER
            diff_row_num = np.any(hardout_bins != codewords_bins)
            '''
            print(f"\ncodewords_poly \t\t\t{codewords_poly}")
            print(f"received_hardDecision_poly \t{received_hardDecision_poly}")
            
            if np.all(received_bins==codewords_bins):
                print("无错误")
            elif np.all(hardout_bins==codewords_bins):
                print("修正，并且修正正确")
            elif np.any(hardout_bins!=codewords_bins):
                print("修正，错误的修正")
            '''

            BLER_thisbatch = diff_row_num / 1.0
            BLER_final = batch_id / (batch_id + 1) * BLER_final + 1 / (batch_id + 1) * BLER_thisbatch
            error_blocks_now += diff_row_num
            # 临时计算(invalidtimeout, validwrong)
            batch_id += 1
            print('\rDiscreteComplexSNR_dB = %.2f      BER = %.3e      BLER = %.3e   (%2d/50)' % (ComplexSNR_dB, BER_final, BLER_final, error_blocks_now), end='', flush=True)
        
        print('\rDiscreteComplexSNR_dB = %.2f      BER = %.3e      BLER = %.3e   (%2d/50)' % (ComplexSNR_dB, BER_final, BLER_final, error_blocks_now), flush=True)

            


