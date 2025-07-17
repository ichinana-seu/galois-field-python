from GF2_map import GF2_map
import numpy as np
from itertools import combinations




# 质因数分解，返回的结果为（质因数，质因数对应的次数）
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

# 判断一个数是否是质数，调用了prime_factor_v2的结果
def is_prime(num):
    """判断num是否为质数"""
    if num <= 1:
        return False  # 1和负数都不是质数
    factors, counts = prime_factor_v2(num)
    # 质数的质因数只有自身，且次数为1
    prime_flag = ( len(factors) == 1 and counts[0] == 1 )
    return prime_flag

# a n1 + b n2 = gcd(n1, n2) 扩展欧几里得算法：返回( gcd(n1, n2), a, b)
def extended_gcd(n1, n2):
    if n2 == 0:
        # 递归终止条件：gcd(a, 0) = a，此时x=1, y=0（a*1 + 0*0 = a）
        return (n1, 1, 0)
    else:
        # 递归计算gcd(b, a mod b)及对应的系数x', y'
        gcd, a_prime, b_prime = extended_gcd(n2, n1 % n2)
        # 反推当前层的系数x, y
        a = b_prime
        b = a_prime - (n1 // n2) * b_prime
        return (gcd, a, b)

# a * n_1 + b * n_2 = 1 扩展欧几里得算法：返回一个可能的(a, b)----------------- One Possible Bezout-Coef-Pair 事实上a、b的通解有很多
def find_bezout_coef_ab(n1, n2): 
    gcd, a, b = extended_gcd(n1, n2)
    assert gcd == 1
    return a, b




########################################################################################################################




# 通过 暴力 求解 x^N - 1 多项式在GF(2)域上的因式。返回一个list
def zhx_factorize_given_N_method1_bruteforce(N, doublecheck: bool=False, verbose: bool=False):
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
    """递归分解多项式GiantPoly，将不可约因式添加到factors"""
    factors = []
    PolyNow = GiantPoly.copy()

    tryDegreeFrom = 1
    while True:
        try:
            one_factor = myGF2.poly_GF2_factorize_once____GF2(PolyNow, verbose=verbose, tryDegreeStartFrom=tryDegreeFrom)       # 一种加速手段：既然前面的degree都尝试过了，不可能是因子，那下次就别尝试了
        except AssertionError:                  # 无法进行分解了，那就退出循环吧
            break

        remain, _ = myGF2.poly_div_euclidmod(PolyNow, one_factor)
        # 一般来说，还是double check一下不可约性
        if doublecheck==True:
            if myGF2.poly_GF2_isIrreducible____GF2(one_factor) == True:
                factors.append(one_factor)
            else:
                raise ValueError("[ERROR] 分解后，因子仍然不是 不可约的")
        if doublecheck==False:
            factors.append(one_factor)
        
        tryDegreeFrom = myGF2.poly_degree(one_factor)
        PolyNow = remain

    factors.append(PolyNow)

    # 再次确保每个因子都是不可分的
    if doublecheck==True:
        for some_factor in factors:
            assert myGF2.poly_GF2_isIrreducible____GF2(some_factor)

    return factors

# 通过 分圆陪集和分裂域 求解 x^N - 1 多项式在GF(2)域上的因式........无法处理重根。返回一个list  【只有非重根时，才能调用该方法】
def zhx_factorize_given_N_method2_splitfield(N):
    if N <= 0 or N % 2 == 0:
        raise ValueError("N必须是正奇数，否则存在重根，method2_splitfield基于分圆陪集和分裂域，无法处理重根")
    
    return 

# 通过 GF(2^m) 求解 x^(2^m-1) - 1 多项式在GF(2)域上的因式........无法处理重根。返回一个list   【只有N = 2^m-1时，才能调用该方法】
def zhx_factorize_given_N_method3_special2pM(N):
    if N <= 0 or N % 2 == 0:
        raise ValueError("N必须是正奇数，否则存在重根，method2_splitfield基于分圆陪集和分裂域，无法处理重根")
    
    return 



# 将一个数分解成两个互素的数 的乘积，返回的结果为0行2列的空结果 或者 x行2列的 x个结果【combination组合数法】
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



def poly_factorize_GF2(poly: np.ndarray, doublecheck: bool=False, verbose: bool=False):
    primitive_polynomial = np.array([1,1,1], dtype=np.int32)
    myGF2 = GF2_map(primitive_polynomial, 2)
    return myGF2.poly_GF2_factorize_all____GF2(poly, doublecheck=doublecheck, verbose=verbose)

def poly_gcd_GF2(polyx: np.ndarray, polyy: np.ndarray):
    primitive_polynomial = np.array([1,1,1], dtype=np.int32)
    myGF2 = GF2_map(primitive_polynomial, 2)
    return myGF2.poly_GF2_poly_gcd____GF2(polyx, polyy)

def poly_lcm_GF2(polyx: np.ndarray, polyy: np.ndarray):
    primitive_polynomial = np.array([1,1,1], dtype=np.int32)
    myGF2 = GF2_map(primitive_polynomial, 2)
    return myGF2.poly_GF2_poly_lcm____GF2(polyx, polyy)





# 尝试将 某个GF(2)上的循环码（例如BCH 255 231）分解为2维的product codes
def zhx_cyclicProductCode_decomposition(n: int, k: int, gx: np.ndarray, verbose: bool=True):
    # 先将码长n分解为互素的两个数的乘积
    coprimePairs = decompose_num_into_coprimePairs(n)

    # 判断x^n - 1 = 0 这个方程组是否含有重根
    # 在GF(2)及其扩域、分裂域中：
    # 1. n为奇数时，p和n互素 -> p不整除n -> 能找到分裂域 -> 无重根               且奇数n分解后依然是两个奇数，非常好
    # 2. n为偶数时，p和n不互素 -> p整除n -> 不能找到分裂域 -> 有重根            【暂时不研究】
    if n % 2 == 0:
        raise NotImplementedError("[ERROR] n为偶数时，p和n不互素 -> p整除n -> 不能找到分裂域 -> 有重根，该功能尚未实现")
    primitive_polynomial = np.array([1,1,1], dtype=np.int32)
    myGF2 = GF2_map(primitive_polynomial, 2)
    # 
    for coprimepair_rowidx in range(0, coprimePairs.shape[0]):
        n1 = coprimePairs[coprimepair_rowidx, 1]
        n2 = coprimePairs[coprimepair_rowidx, 0]
        a,b = find_bezout_coef_ab(n1, n2)

        print(f'正在处理 case {coprimepair_rowidx+1} : n1={n1}, n2={n2}, a={a}, b={b}')
        factorlist_1 = zhx_factorize_given_N_method1_bruteforce(n1)
        factorsetidx_1 = set(range(0, len(factorlist_1)))
        factorlist_2 = zhx_factorize_given_N_method1_bruteforce(n2)
        factorsetidx_2 = set(range(0, len(factorlist_2)))
        print("factorsetidx_1")
        print(factorsetidx_1)
        print("factorsetidx_2")
        print(factorsetidx_2)

        giantpoly = -1 * np.ones([n+1], dtype=np.int32)
        giantpoly[0]=0
        giantpoly[n]=0
        for k1 in range(1, len(factorsetidx_1)):  # set1的选择数量：1
            for k2 in range(1, len(factorsetidx_2)):  # set2的选择数量：1
                print(f"\n    --------------- 从factor1选{k1}个，从factor2选{k2}个 ---------------\n")
                combinations1 = combinations(factorsetidx_1, k1)
                for combo1 in combinations1:
                    combinations2 = combinations(factorsetidx_2, k2)
                    for combo2 in combinations2:
                        # ################################################
                        gx1 = np.array([0], dtype=np.int32)
                        for factoridx in combo1:
                            gx1 = myGF2.poly_mul(gx1, factorlist_1[factoridx])
                        # 将x^{b*n2}代换掉x
                        non_zero_indices = np.where(gx1 == 0)[0]
                        new_degrees = np.array([kkkkk * (b*n2) for kkkkk in non_zero_indices], dtype=np.int32)
                        # 让次数保持在0到n-1区间内
                        new_degrees = new_degrees % n
                        unique_elements, counts = np.unique(new_degrees, return_counts=True)            # 如果出现了偶数个 x^33次方，则可以约去。
                        new_degrees = unique_elements[counts % 2 == 1]                                  # 如果出现了偶数个 x^33次方，则可以约去。
                        # 还原回numpy表示的多项式
                        assert len(new_degrees) > 0 , f"gx1={gx1}    ,   new_degrees={new_degrees} "
                        max_degree = np.max(new_degrees)
                        gx1ext = -1 * np.ones(max_degree + 1, dtype=np.int32)
                        gx1ext[new_degrees] = 0
                        # 计算gcd
                        gcd1 = myGF2.poly_GF2_poly_gcd____GF2(giantpoly, gx1ext)

                        # ################################################
                        gx2 = np.array([0], dtype=np.int32)
                        for factoridx in combo2:
                            gx2 = myGF2.poly_mul(gx2, factorlist_2[factoridx])
                        # 将x^{b*n2}代换掉x
                        non_zero_indices = np.where(gx2 == 0)[0]
                        new_degrees = np.array([kkkkk * (a*n1) for kkkkk in non_zero_indices], dtype=np.int32)
                        # 让次数保持在0到n-1区间内
                        new_degrees = new_degrees % n
                        unique_elements, counts = np.unique(new_degrees, return_counts=True)            # 如果出现了偶数个 x^33次方，则可以约去。
                        new_degrees = unique_elements[counts % 2 == 1]                                  # 如果出现了偶数个 x^33次方，则可以约去。
                        # 还原回numpy表示的多项式
                        assert len(new_degrees) > 0 , f"gx2={gx2}     ,    new_degrees={new_degrees}"
                        max_degree = np.max(new_degrees)
                        gx2ext = -1 * np.ones(max_degree + 1, dtype=np.int32)
                        gx2ext[new_degrees] = 0
                        # 计算gcd
                        gcd2 = myGF2.poly_GF2_poly_gcd____GF2(giantpoly, gx2ext)

                        # ################################################
                        lcm = myGF2.poly_GF2_poly_lcm____GF2(gcd1, gcd2)

                        # ################################################
                        print(f"gx1={gx1} ,gx2={gx2}")
                        print(f"gx1ext={gx1ext} ,gx2ext={gx2ext}")
                        print(f"gcd1={gcd1} ,gcd2={gcd2}")
                        print(f"gx=lcm={lcm}")
                        
                        # 

                exit(0)

        

                    

if __name__ == "__main__":
    # 示例：分解255
    factors, counts = prime_factor_v2(200)
    print(f"质因数: {factors}")
    print(f"对应次数: {counts}")

    print(decompose_num_into_coprimePairs(200))

    
    # factorlist= zhx_factorize_given_N_method1_bruteforce(105, doublecheck=False, verbose=True)
    # print(factorlist)
    # 对比一下，是不是分解全了


    # 
    print(find_bezout_coef_ab(17,15))

    '''
    # 分解x60 + x45 + x30 + 1
    g2polyext = -1 * np.ones([60+1], dtype=np.int32)
    g2polyext[0]=0
    g2polyext[60]=0
    g2polyext[45]=0
    g2polyext[30]=0
    print(poly_factorize_GF2(g2polyext, doublecheck=False, verbose=True))
    '''
    '''
    [[0, 0],
    [0, 0, 0],
    [ 0,  0, -1,  0],
    [ 0,  0, -1, -1,  0],
    [ 0, -1, -1,  0,  0], 
    [ 0, 0, 0, 0, 0], 
    [ 0,  0,  0, -1,  0, -1,  0],
    [ 0,  0, -1,  0,  0,  0, -1, -1, -1,  0, -1, -1,  0],
    [ 0, -1, -1, -1,  0, -1,  0,  0, -1,  0,  0, -1,  0],
    [ 0,  0,  0, -1,  0, -1, -1,  0,  0,  0,  0, -1,  0]]
    '''

    '''
    # 分解x70 + x42 + x14 + 1
    g1polyext = -1 * np.ones([70+1], dtype=np.int32)
    g1polyext[0]=0
    g1polyext[70]=0
    g1polyext[42]=0
    g1polyext[14]=0
    print(poly_factorize_GF2(g1polyext, doublecheck=False, verbose=True))
    '''

    '''
    # 分解x35 + x21 + x7 + 1
    g1result = -1 * np.ones([35+1], dtype=np.int32)
    g1result[0]=0
    g1result[35]=0
    g1result[21]=0
    g1result[7]=0
    print(poly_factorize_GF2(g1result, doublecheck=False, verbose=True))
    '''


    '''
    # 分解x77 + x49 + x35 + 1
    testpoly = -1 * np.ones([77+1], dtype=np.int32)
    testpoly[0]=0
    testpoly[49]=0
    testpoly[35]=0
    testpoly[77]=0
    print(poly_factorize_GF2(testpoly, doublecheck=False, verbose=True))
    '''

    '''
    # x77 + x49 + x35 + 1 和 x^105 -1 的GCD
    gcd1 = -1 * np.ones([77+1], dtype=np.int32)
    gcd1[0]=0
    gcd1[49]=0
    gcd1[35]=0
    gcd1[77]=0

    gcd2 = -1 * np.ones([105+1], dtype=np.int32)
    gcd2[0]=0
    gcd2[105]=0

    gcd_result = poly_gcd_GF2(gcd1, gcd2)
    print(gcd_result)
    print(poly_factorize_GF2(gcd_result, doublecheck=False, verbose=True))
    '''

    zhx_cyclicProductCode_decomposition(255, 231, np.ndarray([1,1]))
