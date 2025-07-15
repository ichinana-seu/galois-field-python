from GF2_map import GF2_map
import numpy as np
from itertools import combinations

def prime_factor_v2(num):
    factors = []
    counts = []
    i = 2
    while i * i <= num:
        if num % i == 0:
            count = 0  # count至少为1，先清楚count的值
            while num % i == 0:
                count += 1
                num = num // i
            factors.append(i)  # 质因数
            counts.append(count)  # 对应次数
        else:
            i += 1

    if num > 1:
        factors.append(num)
        counts.append(1)
    return factors, counts


def is_prime(num):
    """判断num是否为质数"""
    if num <= 1:
        return False  # 1和负数都不是质数
    factors, counts = prime_factor_v2(num)
    # 质数的质因数只有自身，且次数为1
    prime_flag = ( len(factors) == 1 and counts[0] == 1 )
    return prime_flag


def decompose_num_into_coprimePairs(num):
    factors, counts = prime_factor_v2(num)

    # num<= 1 ：    无法·分解
    if num <= 1:
        raise NotImplementedError("小于等于1的数没有有效分解")
        return np.zeros([0,2], dtype=np.int32)  # 小于等于1的数没有有效分解，返回0行2列的空结果
    
    # num为质数 ： 无法分解
    if is_prime(num):
        raise NotImplementedError("num为质数没有有效分解")
        return np.zeros([0,2], dtype=np.int32)  # num为质数没有有效分解，返回0行2列的空结果

    # num不为质数 ,但是 num为单一质数的幂次 ：    则只有(1, n)一种平凡分解，而1不是质数，所以这种情况也不存在
    group_numbers = len(factors)
    if group_numbers == 1:
        raise NotImplementedError("num为单一质数的幂次没有有效分解")
        return np.zeros([0,2], dtype=np.int32)  # num为单一质数的幂次没有有效分解，返回0行2列的空结果
    

    # 正常情况  ：  生成所有非空真子集（用于拆分质因数）
    all_pairs = set()
    # 遍历所有可能的非空子集大小
    for k in range(1, group_numbers):
        # 选择k个质因数的所有组合
        for subset in combinations(range(group_numbers), k):  # 从0到group_numbers-1的范围中，选择3个，并枚举
            # 计算子集对应的乘积a
            a = 1
            for idx in subset:
                a *= factors[idx] ** counts[idx]
            
            # 计算补集对应的乘积b
            b = 1
            for idx in range(group_numbers):
                if idx not in subset:
                    b *= factors[idx] ** counts[idx]
            
            # 确保a < b，避免重复 （不会出现a=b的情况）
            if a <= b:
                all_pairs.add((a, b))
            else:
                all_pairs.add((b, a))
    
    # 转换为numpy数组并排序
    coprimePairs = np.array(sorted(all_pairs), dtype=int)
    return coprimePairs


def zhx_cyclicProductCode_decomposition(n: int, k: int, gx: np.ndarray, verbose: bool=True):
    # 先将码长n分解为互素的两个数的乘积
    coprimePairs = decompose_num_into_coprimePairs(n)

    # 判断x^n - 1 = 0 这个方程组是否含有重根
    # 在GF(2)及其扩域、分裂域中：
    # 1. n为奇数时，p和n互素 -> p不整除n -> 能找到分裂域 -> 无重根
    # 2. n为偶数时，p和n不互素 -> p整除n -> 不能找到分裂域 -> 有重根
    if n % 2 == 0:
        raise NotImplementedError("[ERROR] n为偶数时，p和n不互素 -> p整除n -> 不能找到分裂域 -> 有重根，该功能尚未实现")
    
    # 
    for coprimepair_rowidx in range(0, coprimePairs.shape[0]):
        n1 = coprimePairs[coprimepair_rowidx, 1]
        n2 = coprimePairs[coprimepair_rowidx, 0]

        zhx_factorize_given_N_method1_bruteforce(n1)
        zhx_factorize_given_N_method1_bruteforce(n2)
    return



# 通过暴力求解 x^N - 1 多项式在GF(2)域上的因式。返回一个list
def zhx_factorize_given_N_method1_bruteforce(N):
    """
    暴力求解GF(2)上x^N-1的不可约因式分解
    N为奇数，返回不可约因式的列表（每个因式用系数列表表示，索引对应次数）
    """
    # 由于仅仅用到GF2上运算，就采用GF(2^2)进行平凡运算。
    primitive_polynomial = np.array([1,1,1], dtype=np.int32)
    myGF2 = GF2_map(primitive_polynomial, 2)

    GiantPoly = -1 * np.ones([N+1], dtype=np.int32)
    GiantPoly[0]=myGF2.addinverse(0)
    GiantPoly[N]=0
    print(GiantPoly)
    """递归分解多项式GiantPoly，将不可约因式添加到factors"""
    factors = []
    PolyNow = GiantPoly.copy()
    while myGF2.poly_GF2_isIrreducible____GF2(PolyNow) == False:
        one_factor = myGF2.poly_GF2_factorize_once____GF2(PolyNow)
        remain, _ = myGF2.poly_div_euclidmod(PolyNow, one_factor)
        if myGF2.poly_GF2_isIrreducible____GF2(one_factor) == True:
            factors.append(one_factor)
        PolyNow = remain
    factors.append(remain)

    # 再次确保每个因子都是不可分的
    for some_factor in factors:
        assert myGF2.poly_GF2_isIrreducible____GF2(some_factor)
    return factors



def zhx_factorize_given_N_method2_splitfield(N):
    if N <= 0 or N % 2 == 0:
        raise ValueError("N必须是正奇数，否则存在重根，method2_splitfield基于分圆陪集和分裂域，无法处理重根")
    
    return 


if __name__ == "__main__":
    # 示例：分解255
    factors, counts = prime_factor_v2(255)
    print(f"质因数: {factors}")
    print(f"对应次数: {counts}")

    print(decompose_num_into_coprimePairs(255))

    # zhx_cyclicProductCode_decomposition(255, 231, np.ndarray([1,1]))
    factorlist= zhx_factorize_given_N_method1_bruteforce(15)
    print(factorlist)
    # 对比一下，是不是分解全了
