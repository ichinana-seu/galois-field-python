import os
import numpy as np
import torch
import math

from tqdm import trange     # 进度条循环

from zhx_randomBinary_generator_Numpy import zhx_randomBinary_generator_Numpy
from GF2_map import GF2_map
import zhx_RS_EuclidDecoder



if __name__ == "__main__":
    # 参数
    RS_m = 4
    primitive_polynomial = np.array([1,1,0,0,1], dtype=np.int32)
    myGF2map = GF2_map(primitive_polynomial, RS_m)
    RS_t = 3
    gx = np.array( [ 6 , 9 , 6,  4, 14, 10,  0] , dtype=np.int32)       # RS码生成多项式

    ComplexSNR_dB_set = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    output_prefix = 'ReedSolomon_2p4_t3_N15K9_n60k36-AR0.8'                    # 输出文件夹的名称
    forTestSet_AR_gaussianNoise_para_eta = 0.8

    #训练参数
    line_forEachSNR = 10 * 250

    # 自动提取的参数
    RS_n = 2**RS_m-1
    RS_binaryLength = RS_n*RS_m


    # 实例化编码器、译码器
    myTestGen = zhx_randomBinary_generator_Numpy(RS_n, RS_m, RS_t, gx, myGF2map)
    myTestGen.setpara_coloredGaussianARnoise(forTestSet_AR_gaussianNoise_para_eta)
    # 创建输出的文件夹
    directory = f'./data__{output_prefix}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    outfilename_X1 = f"{directory}/data__{output_prefix}__testX_simpleLLR.npybin"
    outfilename_Y1 = f"{directory}/data__{output_prefix}__testY_simpleColorN.npybin"
    fp_X1 = open(outfilename_X1, mode='ab')
    fp_Y1 = open(outfilename_Y1, mode='ab')

    # 生成噪声（以discrete complex SNR度量，而不是以Eb/N0度量）
    for ik in trange(0, line_forEachSNR, ncols=100): 
        for ComplexSNR_dB in ComplexSNR_dB_set:
            ComplexSNR = 10 ** (ComplexSNR_dB / 10)
            noisepower = 1.0 / ComplexSNR
            noisepower_one_dim = noisepower / 2.0
            sigma_onedim = math.sqrt(noisepower_one_dim)
            codewords_poly = myTestGen.gen_codewords_poly()
            codewords_bins = zhx_RS_EuclidDecoder.zhx_RS_poly2binseq(codewords_poly, myGF2map)
            noise_bins = myTestGen.gen_coloredGaussianAR_noise(sigma_onedim)
            # channel
            received_bins = myTestGen.Bpsk_transfer(codewords_poly, noise_bins)
            # pre-process
            # received_hardDecision = (received_bins<0).astype(np.int32)
            # received_hardDecision_poly = zhx_RS_EuclidDecoder.zhx_RS_binseq2poly(received_hardDecision, myGF2map)
            # hardout_poly, failflag = zhx_RS_EuclidDecoder.zhx_RS_EuclidDecoder(received_hardDecision_poly, RS_t, myGF2map)
            # hardout_bins = zhx_RS_EuclidDecoder.zhx_RS_poly2binseq(hardout_poly, myGF2map)
            
            # 保存结果
            network_X1 = received_bins
            network_Y1 = noise_bins
            print(f"len of network_X1 = {len(network_X1)}")
            print(f"len of network_Y1 = {len(network_Y1)}")
            network_X1.astype(np.float32).tofile(fp_X1)
            network_Y1.astype(np.float32).tofile(fp_Y1)
    fp_X1.close()
    fp_Y1.close()