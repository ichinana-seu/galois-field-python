# version: 1 (2025-05-25)

import numpy as np
import torch
from GF2_map import GF2_map
from scipy.linalg import sqrtm , inv
import zhx_RS_EuclidDecoder_torchBatch


class zhx_randomBinary_generator_torchBatch():
    def __init__(self,  RS_n: int, RS_m: int, RS_t: int, gx: np.ndarray, myGF2map: GF2_map, device: str):
        self.RS_n = RS_n
        self.RS_m = RS_m
        self.RS_t = RS_t
        self.RS_k = RS_n - 2*RS_t
        self.RS_binaryLength = RS_n*RS_m
        assert RS_n == 2**RS_m -1
        self.myGF2map = myGF2map
        self.gx = gx
        self.device = device
        self.para_coloredGaussianARnoise_setflag = False
        self.para_coloredGaussianARnoise_transfer_mat_numpy = None
        self.para_coloredGaussianARnoise_transfer_mat_torch = None
        self.para_coloredGaussianARnoise_INVtransfer_mat_numpy = None
        self.para_coloredGaussianARnoise_INVtransfer_mat_torch = None

    def setpara_coloredGaussianARnoise(self, corr_eta:float):
        N = self.RS_binaryLength
        corrMatrix = np.zeros((N, N))
        for ii in range(N):
            for jj in range(ii, N):
                corrMatrix[ii, jj] = corr_eta ** abs(ii - jj)
                corrMatrix[jj, ii] = corrMatrix[ii, jj]
        transfer_mat = sqrtm(corrMatrix)
        assert isinstance(transfer_mat, np.ndarray)  # 确保transfer_mat是一个正常运行的np.ndarray对象（保证其为True）
        self.para_coloredGaussianARnoise_transfer_mat_numpy = transfer_mat
        self.para_coloredGaussianARnoise_transfer_mat_torch = torch.from_numpy(self.para_coloredGaussianARnoise_transfer_mat_numpy).to(device=self.device, dtype=torch.float32)
        # 白化
        invtransfer_mat = inv(transfer_mat)
        self.para_coloredGaussianARnoise_INVtransfer_mat_numpy = invtransfer_mat
        self.para_coloredGaussianARnoise_INVtransfer_mat_torch = torch.from_numpy(self.para_coloredGaussianARnoise_INVtransfer_mat_numpy).to(device=self.device, dtype=torch.float32)
        self.para_coloredGaussianARnoise_setflag = True


    def gen_awgn_noise(self, sigma: float, batchsize: int):
        std_gaussian_noise = torch.randn( (batchsize, self.RS_binaryLength) , dtype=torch.float32, device=self.device)
        return sigma * std_gaussian_noise
    
    def gen_coloredGaussianAR_noise(self, sigma: float, batchsize: int):
        if self.para_coloredGaussianARnoise_setflag == False:
            print('[ERROR] Set AR noise parameter first!')
            exit(1)
        # 生成标准高斯噪声，AWGN噪声，ACGN(AR)噪声
        std_gaussian_noise = torch.randn( (batchsize, self.RS_binaryLength) , dtype=torch.float32, device=self.device)
        white_gaussian_noise = sigma * std_gaussian_noise
        color_gaussian_noise = torch.matmul(white_gaussian_noise, self.para_coloredGaussianARnoise_transfer_mat_torch)
        return color_gaussian_noise

    def convert_awgn_to_GaussianAR(self, white_gaussian_noise: torch.Tensor): 
        if self.para_coloredGaussianARnoise_setflag == False:
            print('[ERROR] Set AR noise parameter first!')
            exit(1)
        color_gaussian_noise = torch.matmul(white_gaussian_noise, self.para_coloredGaussianARnoise_transfer_mat_torch)
        return color_gaussian_noise
    
    def convert_GaussianAR_to_awgn_WHITELIZE(self, color_gaussian_noise: torch.Tensor):
        if self.para_coloredGaussianARnoise_setflag == False:
            print('[ERROR] Set AR noise parameter first!')
            exit(1)
        white_gaussian_noise = torch.matmul(color_gaussian_noise, self.para_coloredGaussianARnoise_INVtransfer_mat_torch)
        return white_gaussian_noise
    
    def gen_codewords_poly(self, batchsize: int):
        info_vector_poly = np.random.randint(-1,self.RS_n, size=[batchsize, self.RS_k]).astype(np.int32)
        codewords_poly = (-1) * np.ones([batchsize, self.RS_n], dtype=np.int32)
        for row in range(0, batchsize):
            tmp = self.myGF2map.poly_mul( self.myGF2map.poly_fresh(self.gx) , self.myGF2map.poly_fresh(info_vector_poly[row, :]) ) 
            codewords_poly[row, 0:len(tmp)] = tmp
        codewords_poly_torch = torch.from_numpy(codewords_poly).to(device=self.device, dtype=torch.int32)
        return codewords_poly_torch
    
    def gen_codewords_poly_and_bins(self, batchsize: int):
        info_vector_poly = np.random.randint(-1,self.RS_n, size=[batchsize, self.RS_k]).astype(np.int32)
        codewords_poly = (-1) * np.ones([batchsize, self.RS_n], dtype=np.int32)
        codewords_bins = (-1) * np.ones([batchsize, self.RS_binaryLength], dtype=np.int32)
        for row in range(0, batchsize):
            tmp = self.myGF2map.poly_mul( self.myGF2map.poly_fresh(self.gx) , self.myGF2map.poly_fresh(info_vector_poly[row, :]) ) 
            codewords_bins[row, :] = zhx_RS_EuclidDecoder_torchBatch.zhx_RS_poly2binseq(tmp, self.myGF2map)
            codewords_poly[row, 0:len(tmp)] = tmp
        codewords_poly_torch = torch.from_numpy(codewords_poly).to(device=self.device, dtype=torch.int32)
        codewords_bins_torch = torch.from_numpy(codewords_bins).to(device=self.device, dtype=torch.int32)
        return codewords_poly_torch, codewords_bins_torch


    def Bpsk_transfer(self, codewords_bins_torch: torch.Tensor, noise_bins: torch.Tensor):
        assert codewords_bins_torch.shape == noise_bins.shape
        x_transmitted = (1 - 2 * codewords_bins_torch)
        y_received = x_transmitted + noise_bins
        return y_received
    
