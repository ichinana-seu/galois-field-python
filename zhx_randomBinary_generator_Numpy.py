# version: 1 (2025-05-20)

import numpy as np
from GF2_map import GF2_map
from scipy.linalg import sqrtm , inv
import zhx_RS_EuclidDecoder

class zhx_randomBinary_generator_Numpy():
    def __init__(self,  RS_n: int, RS_m: int, RS_t: int, gx: np.ndarray, myGF2map: GF2_map):
        self.RS_n = RS_n
        self.RS_m = RS_m
        self.RS_t = RS_t
        self.RS_k = RS_n - 2*RS_t
        self.RS_binaryLength = RS_n*RS_m
        assert RS_n == 2**RS_m -1
        self.myGF2map = myGF2map
        self.gx = gx
        self.para_coloredGaussianARnoise_transfer_mat_numpy = None
        self.para_coloredGaussianARnoise_INVtransfer_mat_numpy = None

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
        # 白化
        invtransfer_mat = inv(transfer_mat)
        self.para_coloredGaussianARnoise_INVtransfer_mat_numpy = invtransfer_mat


    def gen_awgn_noise(self, sigma: float):
        std_gaussian_noise = np.random.randn(self.RS_binaryLength).astype(np.float32)
        return sigma * std_gaussian_noise
    
    def gen_coloredGaussianAR_noise(self, sigma: float):
        if self.para_coloredGaussianARnoise_transfer_mat_numpy == None:
            print('[ERROR] Set AR noise parameter first!')
            exit(1)
        # 生成标准高斯噪声，AWGN噪声，ACGN(AR)噪声
        std_gaussian_noise = np.random.randn(self.RS_binaryLength).astype(np.float32)
        white_gaussian_noise = sigma * std_gaussian_noise
        color_gaussian_noise = np.matmul(white_gaussian_noise, self.para_coloredGaussianARnoise_transfer_mat_numpy)
        return color_gaussian_noise

    def convert_awgn_to_GaussianAR(self, white_gaussian_noise): 
        if self.para_coloredGaussianARnoise_transfer_mat_numpy == None:
            print('[ERROR] Set AR noise parameter first!')
            exit(1)
        color_gaussian_noise = np.matmul(white_gaussian_noise, self.para_coloredGaussianARnoise_transfer_mat_numpy)
        return color_gaussian_noise
    
    def convert_GaussianAR_to_awgn_WHITELIZE(self, color_gaussian_noise):
        if self.para_coloredGaussianARnoise_INVtransfer_mat_numpy == None:
            print('[ERROR] Set AR noise parameter first!')
            exit(1)
        white_gaussian_noise = np.matmul(color_gaussian_noise, self.para_coloredGaussianARnoise_INVtransfer_mat_numpy)
        return white_gaussian_noise
    
    def gen_codewords_poly(self):
        info_vector_poly = np.random.randint(-1,self.RS_n, size=[self.RS_k]).astype(np.int32)
        codewords_poly = self.myGF2map.poly_mul( self.myGF2map.poly_fresh(self.gx) , self.myGF2map.poly_fresh(info_vector_poly) ) 
        return codewords_poly
    
    def Bpsk_transfer(self, codewords_poly: np.ndarray, noise_bins: np.ndarray):
        codewords_bins = zhx_RS_EuclidDecoder.zhx_RS_poly2binseq(codewords_poly, self.myGF2map)
        x_transmitted = (1 - 2 * codewords_bins)
        y_received = x_transmitted + noise_bins
        return y_received
    
