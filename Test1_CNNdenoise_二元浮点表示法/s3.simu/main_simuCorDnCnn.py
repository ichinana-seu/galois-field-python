import os
import numpy as np
import torch
import math

from tqdm import trange     # 进度条循环

from zhx_randomBinary_generator_Numpy import zhx_randomBinary_generator_Numpy
from GF2_map import GF2_map
import zhx_RS_EuclidDecoder

from myConvNet_CorDnCNN import CorDnCNN

if __name__ == "__main__":
    # 参数
    RS_m = 4
    primitive_polynomial = np.array([1,1,0,0,1], dtype=np.int32)
    myGF2map = GF2_map(primitive_polynomial, RS_m)
    RS_t = 3
    gx = np.array( [ 6 , 9 , 6,  4, 14, 10,  0] , dtype=np.int32)       # RS码生成多项式

    # model
    Eval_ComplexSNR_dB_set = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    IO_prefix = 'ReedSolomon_2p4_t3_N15K9_n60k36-AR0.8'
    AR_gaussianNoise_para_eta = 0.8
    device = 'cuda:0'

    # 自动提取的参数
    RS_n = 2**RS_m-1
    RS_binaryLength = RS_n*RS_m

    INPUT_model_directory = f'./model__{IO_prefix}'
    if not os.path.exists(INPUT_model_directory):
        os.makedirs(INPUT_model_directory)
    OUTPUT_simu_directory = f'./simu__{IO_prefix}'
    if not os.path.exists(OUTPUT_simu_directory):
        os.makedirs(OUTPUT_simu_directory)
    OUTPUT_simu_filename = f"{OUTPUT_simu_directory}/simu__{IO_prefix}__results.txt"


    # 实例化编码器、译码器
    myTestGen = zhx_randomBinary_generator_Numpy(RS_n, RS_m, RS_t, gx, myGF2map)
    myTestGen.setpara_coloredGaussianARnoise(AR_gaussianNoise_para_eta)

    model_filename = f"{INPUT_model_directory}/model__{IO_prefix}__simpleTestCorDnCNN.pth"
    myNet = torch.load(model_filename, map_location=torch.device(device))

    # Example
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
            noise_bins = myTestGen.gen_coloredGaussianAR_noise(sigma_onedim)
            # channel
            received_bins = myTestGen.Bpsk_transfer(codewords_poly, noise_bins)
            # decode - denoise
            received_bins_torch = torch.from_numpy(received_bins).to(device=device, dtype=torch.float32)
            est_noise_torch = myNet(received_bins_torch)
            received_denoised_torch = received_bins_torch - est_noise_torch
            received_denoised = received_denoised_torch.cpu().numpy()
            received_hardDecision = (received_denoised<0).astype(np.int32)
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
            BLER_thisbatch = diff_row_num / 1.0
            BLER_final = batch_id / (batch_id + 1) * BLER_final + 1 / (batch_id + 1) * BLER_thisbatch
            error_blocks_now += diff_row_num
            # 临时计算(invalidtimeout, validwrong)
            batch_id += 1

            print('\rDiscreteComplexSNR_dB = %.2f      BER = %.3e      BLER = %.3e   (%2d/50)' % (ComplexSNR_dB, BER_final, BLER_final, error_blocks_now), end='', flush=True)
        
        print('\rDiscreteComplexSNR_dB = %.2f      BER = %.3e      BLER = %.3e   (%2d/50)' % (ComplexSNR_dB, BER_final, BLER_final, error_blocks_now), flush=True) 
        with open(OUTPUT_simu_filename, mode='a') as fp:
            fp.write('DiscreteComplexSNR_dB = %.2f      BER = %.3e      BLER = %.3e \n' % (ComplexSNR_dB, BER_final, BLER_final))
    with open(OUTPUT_simu_filename, mode='a') as fp:
        fp.write('\n\n')