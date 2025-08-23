from GF2_map import GF2_map
import numpy as np
from itertools import combinations














########################### 以下为测试 ##########################
if __name__ == "__main__":
    print("Reticulating splines...")
    # 测试function1，排序与范德蒙德逆矩阵
    m = 4
    primitive_poly = np.array([1,0,0,1,1], dtype=np.int32)
    myGF2 = GF2_map(primitive_poly, m)
    
    LLR_rec = np.random.
