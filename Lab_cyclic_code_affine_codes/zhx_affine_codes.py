from GF2_map import GF2_map
import numpy as np
from itertools import combinations





def zhx_genFootscriptPairs(LLR_rec: np.ndarray, myGF2map: GF2_map, verbose: bool=False):
    assert len(LLR_rec) == 2 ** myGF2map.m
    LLR_abs = np.abs(LLR_rec)
    LLR_sorted_argidx = np.argsort(-LLR_abs)
    LLR_sorted = LLR_abs[LLR_sorted_argidx]
    foot_ordinates_ORIGIN = np.arange(start=-1, stop=N-1, step=1).astype(np.int32)
    foot_ordinates_TRANSED = foot_ordinates_ORIGIN[LLR_sorted_argidx]
    if verbose==True:
        print("[INFO] LLR_abs")
        print(LLR_abs)
        print("[INFO] LLR_sorted")
        print(LLR_sorted)
        print("[INFO] LLR_argsort_idx")
        print(LLR_sorted_argidx)
        print("[INFO] transform_pair_ORIGIN")
        print(foot_ordinates_ORIGIN)
        print("[INFO] transform_pair_TRANSED")
        print(foot_ordinates_TRANSED)
    return LLR_sorted_argidx, LLR_abs, foot_ordinates_ORIGIN, LLR_sorted, foot_ordinates_TRANSED


def zhx_calCoefficientsOfPermutationFunction(Order_m_dividedby_e: int, foot_ordinates_ORIGIN: np.ndarray, foot_ordinates_TRANSED: np.ndarray, verbose: bool=False):
    footscript_ORIGIN = foot_ordinates_ORIGIN[1:Order_m_dividedby_e+1]
    footscript_TRANSED = foot_ordinates_TRANSED[1:Order_m_dividedby_e+1]
    coef_b = foot_ordinates_TRANSED[0]
    if verbose==True:
        print("[INFO] footscript_ORIGIN")
        print(footscript_ORIGIN)
        print("[INFO] footscript_TRANSED")
        print(footscript_TRANSED)
        print("coef_b")
        print(coef_b)



########################### 以下为测试 ##########################
if __name__ == "__main__":
    print("Reticulating splines...")
    # 测试function1，排序与范德蒙德逆矩阵
    m = 4
    primitive_poly = np.array([1,0,0,1,1], dtype=np.int32)
    myGF2 = GF2_map(primitive_poly, m)
    N = 2**m
    sigma = 0.1
    info = np.random.randint(0,2, size=[N]).astype(np.int32)
    bpsk = 1 - 2*info
    noise = np.random.normal(loc=0.0, scale=sigma, size=N).astype(np.float32)
    LLR_rec = bpsk + noise
    LLR_sorted_argidx, LLR_abs, foot_ordinates_ORIGIN, LLR_sorted, foot_ordinates_TRANSED = zhx_genFootscriptPairs(LLR_rec, myGF2, verbose=True)
    zhx_calCoefficientsOfPermutationFunction(2, foot_ordinates_ORIGIN, foot_ordinates_TRANSED, verbose=True)
    
