import numpy as np
import torch
import os
import math

from zhx_randomBinary_generator_torchBatch import zhx_randomBinary_generator_torchBatch
from GF2_map import GF2_map
import zhx_RS_EuclidDecoder_torchBatch


if __name__ == "__main__":
    device = 'xpu'
    # 
    batchsize = 1000
    gx = np.array( [ 6 , 9 , 6,  4, 14, 10,  0] , dtype=np.int32)
    RS_m = 4
    RS_t = 3
    RS_n = 2**RS_m-1
    RS_binaryLength = RS_n*RS_m

    AR_eta = 0.8

    primitive_polynomial = np.array([1,1,0,0,1], dtype=np.int32)
    myGF2map = GF2_map(primitive_polynomial, RS_m)
    myTestGen = zhx_randomBinary_generator_torchBatch(RS_n, RS_m, RS_t,gx, myGF2map, device)
    myTestGen.setpara_coloredGaussianARnoise(AR_eta)
    

    Eval_ComplexSNR_dB_set = [4.0]
    for ComplexSNR_dB in Eval_ComplexSNR_dB_set: 
        batch_id = 0
        error_blocks_now = 0
        BER_final = 1.0
        BLER_final = 1.0

        # transmit-paras
        ComplexSNR = 10 ** (ComplexSNR_dB / 10)
        noisepower = 1.0 / ComplexSNR
        noisepower_one_dim = noisepower / 2.0
        sigma_onedim = math.sqrt(noisepower_one_dim)
        while error_blocks_now < 50:
            codewords_poly, codewords_bins = myTestGen.gen_codewords_poly_and_bins(batchsize)
            noise_bins = myTestGen.gen_coloredGaussianAR_noise(sigma_onedim, batchsize)
            # channel
            received_bins = myTestGen.Bpsk_transfer(codewords_bins, noise_bins)
            # pre-process
            received_hardDecision = (received_bins<0).to(torch.int32)
            hardout_bins = zhx_RS_EuclidDecoder_torchBatch.zhx_RS_EuclidDecoder_inputBins_torchBatch(received_hardDecision, RS_t, myGF2map, device)
            # 统计BER
            diff_bits = (hardout_bins != codewords_bins)
            diff_bits_num = torch.count_nonzero(diff_bits)
            BER_thisbatch = diff_bits_num / (batchsize * RS_binaryLength)
            BER_final = batch_id / (batch_id + 1) * BER_final + 1 / (batch_id + 1) * BER_thisbatch
            # 统计BLER
            diff_row = torch.any(hardout_bins != codewords_bins, dim=1)
            diff_row_num = torch.count_nonzero(diff_row)
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
            BLER_thisbatch = diff_row_num / batchsize
            BLER_final = batch_id / (batch_id + 1) * BLER_final + 1 / (batch_id + 1) * BLER_thisbatch
            error_blocks_now += diff_row_num
            # 临时计算(invalidtimeout, validwrong)
            batch_id += 1
            print('\rDiscreteComplexSNR_dB = %.2f      BER = %.3e      BLER = %.3e   (%2d/50)' % (ComplexSNR_dB, BER_final, BLER_final, error_blocks_now), end='', flush=True)
        
        print('\rDiscreteComplexSNR_dB = %.2f      BER = %.3e      BLER = %.3e   (%2d/50)' % (ComplexSNR_dB, BER_final, BLER_final, error_blocks_now), flush=True)

            


