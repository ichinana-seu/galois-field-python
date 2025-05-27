import os
import numpy as np
import torch
import math

from tqdm import trange     # 进度条循环

from zhx_randomBinary_generator_torchBatch import zhx_randomBinary_generator_torchBatch
from GF2_map import GF2_map
import zhx_RS_EuclidDecoder_torchBatch

from myConvNet_CorDnCNN import CorDnCNN

if __name__ == "__main__":
    batchsize = 1000
    device = 'xpu'
    # 参数
    RS_m = 4
    primitive_polynomial = np.array([1,1,0,0,1], dtype=np.int32)
    myGF2map = GF2_map(primitive_polynomial, RS_m)
    RS_t = 3
    gx = np.array( [ 6 , 9 , 6,  4, 14, 10,  0] , dtype=np.int32)       # RS码生成多项式

    # model
    Eval_ComplexSNR_dB_set = [5.0]
    IO_prefix = 'ReedSolomon_2p4_t3_N15K9_n60k36-AR0.8'
    AR_gaussianNoise_para_eta = 0.8

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
    myTestGen = zhx_randomBinary_generator_torchBatch(RS_n, RS_m, RS_t, gx, myGF2map, device)
    myTestGen.setpara_coloredGaussianARnoise(AR_gaussianNoise_para_eta)

    model_filename = f"{INPUT_model_directory}/model__{IO_prefix}__simpleTestCorDnCNN.pth"
    myNet = torch.load(model_filename, map_location=torch.device(device), weights_only=False)
    myNet.eval()

    # Example
    for ComplexSNR_dB in Eval_ComplexSNR_dB_set:
        batch_id = 0
        error_blocks_now = 0
        BER_final = 1.0
        BLER_final = 1.0
        # para
        ComplexSNR = 10 ** (ComplexSNR_dB / 10)
        noisepower = 1.0 / ComplexSNR
        noisepower_one_dim = noisepower / 2.0
        sigma_onedim = math.sqrt(noisepower_one_dim)
        while error_blocks_now < 50:
            codewords_poly, codewords_bins = myTestGen.gen_codewords_poly_and_bins(batchsize)
            noise_bins = myTestGen.gen_coloredGaussianAR_noise(sigma_onedim, batchsize)
            # channel
            received_bins = myTestGen.Bpsk_transfer(codewords_bins, noise_bins)
            # NET
            est_noise_torch = myNet(received_bins)
            received_denoised = received_bins - est_noise_torch
            # pre-process
            received_hardDecision = (received_denoised<0).to(torch.int32)
            hardout_bins = zhx_RS_EuclidDecoder_torchBatch.zhx_RS_EuclidDecoder_inputBins_torchBatch(received_hardDecision, RS_t, myGF2map, device)
            # 统计BER
            diff_bits = (hardout_bins != codewords_bins)
            diff_bits_num = torch.count_nonzero(diff_bits)
            BER_thisbatch = diff_bits_num / (batchsize * RS_binaryLength)
            BER_final = batch_id / (batch_id + 1) * BER_final + 1 / (batch_id + 1) * BER_thisbatch
            # 统计BLER
            diff_row = torch.any(hardout_bins != codewords_bins, dim=1)
            diff_row_num = torch.count_nonzero(diff_row)
            BLER_thisbatch = diff_row_num / batchsize
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