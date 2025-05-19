# version: 6 (2025-05-19)
# 适用于 GF(2^m) 的Galois扩域。请注意：这里的基域只能是2。
# 不可以是其他素数GF(p)->GF(p^m)或者GF(2^n)->GF(2^n^m)
# 表示法：幂次表示法

import numpy as np

# GF 2^2        Appendix        p(X) = 1 + X + X^2                  [1,1,1]
# GF 2^3        Appendix        p(X) = 1 + X^2 + X^3                [1,0,1,1]
# GF 2^4        Appendix        p(X) = 1 + X^3 + X^4                [1,0,0,1,1]
# GF 2^4        Book(ShuLin)    p(X) = 1 + X + X^4                  [1,1,0,0,1]
# GF 2^5        Appendix        p(X) = 1 + X^3 + X^5                [1,0,0,1,0,1]
# GF 2^6        Appendix        p(X) = 1 + X^5 + X^6                [1,0,0,0,0,1,1]
# GF 2^7        Appendix        p(X) = 1 + X^6 + X^7                [1,0,0,0,0,0,1,1]
# GF 2^8        Appendix        p(X) = 1 + X + X^6 + X^7 + X^8      [1,1,0,0,0,0,1,1,1]
# GF 2^9        Appendix        p(X) = 1 + X^5 + X^9                [1,0,0,0,0,1,0,0,0,1]
# GF 2^10       Appendix        p(X) = 1 + X^5 + X^10               [1,0,0,0,0,0,0,1,0,0,1]


class GF2_map():
    # 可访问元素：m, primitive_polynomials, table_exp2tuple, table_tupleInt2exp
    def __init__(self, primitive_polynomials: np.ndarray, m: int):
        assert primitive_polynomials.ndim == 1
        assert primitive_polynomials.dtype == np.int32
        assert len(primitive_polynomials)-1 == m
        self.primitive_polynomials = primitive_polynomials
        self.m = m
        # generate table_exp2tuple (a pipeline way)
        self.table_exp2tuple = np.zeros((2**m, m), dtype=np.int32)
        self.table_exp2tuple[1,0] = 1
        for line in range(2, 2**m):
            self.table_exp2tuple[line, 0] = self.table_exp2tuple[line-1, m-1]
            self.table_exp2tuple[line, 1:m] = ( (primitive_polynomials[1:m] * self.table_exp2tuple[line-1, m-1] ) + self.table_exp2tuple[line-1, 0:m-1]   )%2
        # generate table_tupleInt2exp
        self.table_tupleInt2exp = np.zeros((2**m), dtype=np.int32)
        for line in range(0, 2**m):
            tupleInt = self.convert_tuple2tupleInt(self.table_exp2tuple[line, :])
            exp = line-1
            self.table_tupleInt2exp[tupleInt] = exp
        print("Successfully initiatized GF2_map")

    # 这些函数只在init时用到
    def convert_tuple2tupleInt(self, tuple: np.ndarray):
        return np.sum(tuple * (2 ** np.arange(len(tuple)))).item()
    
    def convert_tupleInt2exp(self, tupleInt: int):
        exp = self.table_tupleInt2exp[tupleInt].item()
        return exp
    
    def convert_exp2tuple(self, x: int):
        assert x>=-1 and x<=2**self.m-2
        find_index = x+1
        return self.table_exp2tuple[find_index, :].copy()
        
    def convert_tuple2exp(self, tuple: np.ndarray):
        tupleInt = self.convert_tuple2tupleInt(tuple)
        exp = self.convert_tupleInt2exp(tupleInt)
        return exp


    # 元素运算
    def add(self, x: int, y: int):
        assert x>=-1 and x<=2**self.m-2
        assert y>=-1 and y<=2**self.m-2
        x_tuple = self.convert_exp2tuple(x)
        y_tuple = self.convert_exp2tuple(y)
        result_tuple = (x_tuple + y_tuple) %2
        result_exp = self.convert_tuple2exp(result_tuple)
        return result_exp

    def mul(self, x: int, y: int):
        assert x>=-1 and x<=2**self.m-2
        assert y>=-1 and y<=2**self.m-2
        if x==-1 or y==-1:
            results = -1
        else:
            results = (x+y) % (2**self.m-1)
        return results

    def addinverse(self, x: int):
        assert x>=-1 and x<=2**self.m-2
        return x

    def mulinverse(self, x: int):
        assert x>=-1 and x<=2**self.m-2
        if x==-1:
            print('[ERROR] alpha^-1 cannot take inverse')
            exit(1)
        result = (2**self.m-1 -x) % (2**self.m-1)
        return result

    # 实用（相当于连乘）
    def pow(self, x: int, power: int):                           # power 个 x 相乘 % 请注意：在该函数中，-1不表示 "alpha^-1 ="0， 而表示 "alpha^(n-1) "
        assert x>=-1 and x<=2**self.m-2
        if x==-1:
            result = -1
        else:
            result = (    (x % (2**self.m-1)  ) * (power % (2**self.m-1) )     ) % (2**self.m-1)         # warning("Especially on 'base_gf2_alpha_pow': alpha^(-1) may be regarded as alpha^(n-1) ")
        return result

    # 实用（相当于连加）
    def addadd(self, x: int, times: int):
        assert times>=1
        assert x>=-1 and x<=2**self.m-2
        result = -1
        for i in range(0,times):
            result = self.add(result, x)
        return result



    # 域上多项式环，其系数应该是 GF(2^m) 中的元素，用指数形式(alpha^i)的i表示。
    def poly_add(self, polyx: np.ndarray, polyy: np.ndarray):
        pow_x = len(polyx)-1
        pow_y = len(polyy)-1
        pow_z = max(pow_x, pow_y)
        long_arrayx = (-1) * np.ones( (pow_z+1 ), dtype=np.int32)
        long_arrayy = (-1) * np.ones( (pow_z+1 ), dtype=np.int32)
        long_arrayx[0: pow_x+1] = polyx
        long_arrayy[0: pow_y+1] = polyy
        vectorized_add_function = np.vectorize(self.add)
        result_arrayz = vectorized_add_function(long_arrayx, long_arrayy)
        result_arrayz = self.poly_fresh(result_arrayz)
        return result_arrayz

    def poly_mul(self, polyx: np.ndarray, polyy: np.ndarray):
        pow_x = len(polyx)-1
        pow_y = len(polyy)-1
        result_arrayz = (-1) * np.ones( (pow_x+pow_y+1), dtype=np.int32 );
        for k in range(0, pow_x+pow_y+1):
            for i in range(max(0,k-pow_y), min(k,pow_x)+1):
                tmp = self.mul(polyx[i], polyy[k-i]);               # 根据卷积，先做乘法
                result_arrayz[k] = self.add(result_arrayz[k], tmp)      # 累加起来
        result_arrayz = self.poly_fresh(result_arrayz)
        return result_arrayz
    
    def poly_addinverse(self, polyx: np.ndarray):            # 可以证明，poly的加逆 是 各自系数的加逆
        vectorized_addinverse_function = np.vectorize(self.addinverse)
        result = vectorized_addinverse_function(polyx)
        result = self.poly_fresh(result)
        return result

    def poly_fresh(self, polyx: np.ndarray):                 # 为了防止最高位0占用空间，可以刷新一下多项式
        pow_x = len(polyx)-1
        results = polyx.copy()
        while(1):
            if results[len(results)-1]==-1 and len(results)!=1:
                results = results[0: len(results)-1]
            else:
                break
        return results
    
    def poly_div_euclidmod(self, polyx: np.ndarray, polyy: np.ndarray):        # 域上多项式环 的欧几里得带余除法
        polyx = self.poly_fresh(polyx)
        polyy = self.poly_fresh(polyy)
        # 处理特殊情况：除数为零多项式
        if np.all(polyy == -1):
            raise ValueError("[ERROR] Divisor cannot be zero-polynomial. 除数不能为零多项式")
        # 处理特殊情况：被除数为零多项式
        if np.all(polyx == -1):
            return np.array([-1]), np.array([-1])
        # 获取多项式次数
        deg_dividend = len(polyx) - 1
        deg_divisor = len(polyy) - 1
        deg_q = deg_dividend - deg_divisor
        deg_r = deg_divisor - 1
        # 处理特殊情况：如果被除数次数小于除数次数，直接返回
        if deg_dividend < deg_divisor:
            return np.array([-1]), polyx.copy()
        # 初始化商和余数
        quotient = (-1) * np.ones(deg_q + 1, dtype=np.int32)
        remainder = polyx.copy()
        while len(remainder)-1 >= deg_divisor:
            # 获取当前余数的最高次项指数和系数
            current_deg = len(remainder) - 1
            current_coeff = remainder[-1]
            # 如果最高次项系数为0（-1表示），跳过此次循环
            if current_coeff == -1:
                remainder = self.poly_fresh(remainder)
                continue
            # 计算商的当前项：current_coeff / divisor的最高次项系数
            divisor_leading_coeff = polyy[-1]
            if divisor_leading_coeff == -1:
                raise ValueError("[ERROR] Divisor cannot be zero-polynomial. 除数不能为零多项式")
            # 计算系数的商（在GF(2^m)中为乘法逆元）
            coeff_quotient = self.mul(current_coeff, self.mulinverse(divisor_leading_coeff))
            # 计算当前项在商中的位置
            quotient_pos = current_deg - deg_divisor
            # 更新商
            quotient[quotient_pos] = coeff_quotient
            # 计算当前项乘以除数
            term = (-1) * np.ones(quotient_pos + len(polyy), dtype=np.int32)
            term[quotient_pos:] = polyy
            term = self.poly_mul(np.array([coeff_quotient]), term)
            # 从余数中减去当前项
            remainder = self.poly_add(remainder, self.poly_addinverse(term))

        quotient = self.poly_fresh(quotient)
        remainder = self.poly_fresh(remainder)
        return quotient, remainder

    # 求出函数值，GF（2^m）中的元素alphas_exp带入f(x)，求出值
    def poly_function_value(self, polyfx: np.ndarray, ele: int):
        results = -1
        results = self.add(results, polyfx[0])
        for process_x_exp in range(1, len(polyfx) ):
            tmp = self.pow(ele, process_x_exp)
            tmp = self.mul(tmp, polyfx[process_x_exp])
            results = self.add(results, tmp)
        return results

    # 求出多项式函数的导数
    def poly_function_derivative(self, polyfx: np.ndarray):
        if len(polyfx) == 1:        # 对常数函数求导，得到的结果是0，也就是-1
            fy = np.array([-1], dtype=np.int32)
            return fy
        fy = polyfx[1:len(polyfx)].copy()
        for y_index in range(0, len(fy) ):
            fy[y_index] = self.addadd(fy[y_index], y_index+1)
        fy = self.poly_fresh(fy)
        return fy

    # mattsonSolomon 变换（Qin Huang 导数码论文形式）
    def poly_mattsonSolomon_transform(self, polyx: np.ndarray):
        n = len(polyx)
        deg = len(polyx) -1
        if n != 2**self.m-1:
            raise NotImplementedError("[ERROR] MS transform: Degree of polynomial must be 2^m-2 (Length of polynomial must be 2^m-1).")
        result_Az_vector = (-1) * np.ones(n, dtype=np.int32)
        for j in range(0, n):
            for i in range(0, n):
                tmp = (-j * i) % n                 # 注意，例如计算( alpha^(-1)^ 3)，这里尽量不要直接调用alpha_pow函数，因为-1会被当作加法零元。
                tmp = self.mul(polyx[i], tmp)
                result_Az_vector[j] = self.add(result_Az_vector[j] , tmp)
        return result_Az_vector
    
    # mattsonSolomon 逆变换（Qin Huang 导数码论文形式）
    def poly_mattsonSolomon_inverseTransform(self, polyX: np.ndarray):
        n = len(polyX)
        deg = len(polyX) -1
        if n != 2**self.m-1:
            raise NotImplementedError("[ERROR] MS transform: Degree of polynomial must be 2^m-2 (Length of polynomial must be 2^m-1).")
        result_az_vector = (-1) * np.ones(n, dtype=np.int32)
        for j in range(0, n):
            result_az_vector[j] = self.poly_function_value(polyX, j)
        return result_az_vector

    # 其他功能：查询元素的阶
    def order_of_element(self, x: int):
        assert x>=-1 and x<=2**self.m-2
        if x == -1:
            print('[ERROR] alpha^(-1) \' order is N/A')
            exit(0)
        
        cnt = 0
        for i in range(1, 2**self.m):
            ans = (x*i) % (2**self.m - 1)
            cnt = cnt+1
            if ans ==0:
                break
        assert (2**self.m-1) % cnt == 0
        return cnt
    
    # 其他功能：打印 所有元素的阶
    def print_elements_order(self):
        print("alpha^(-1)      ( order of this element: N/A )")
        for ele in range(0, 2**self.m-1):
            cnt=0
            print("alpha^(%d) [" % ele ,end='')
            for i in range(1, 2**self.m):
                ans = (ele*i) % (2**self.m - 1)
                cnt = cnt+1
                print("%d, "%ans, end='')
                if ans ==0:
                    break
                
            print(f"]      ( order of this element: {cnt} )  ")
        print("")

    # 其他功能：打印 所有元素的共轭对
    def print_elements_conjugates(self):
        already_record = []
        for ele in range(-1, 2**self.m-1):
            e = 1                               # 该元素的conjugates的数量
            store = []
            while True:
                if self.pow(ele,2**e) == ele:
                    break
                store.append( self.pow(ele,2**e) )
                e = e + 1
            store.insert(0, ele)
            assert type(store)==list
            if (ele not in already_record):
                already_record.extend(store)
                print('alpha^%d has %d conjugates:    ' % (ele,e), end="")
                print(store)
        print('All elements in the same conjugate set constitute a minimal polynomial')
        print("")

    # 其他功能：打印 所有分圆陪集
    def print_elements_cyclotomicCoset(self):
        already_record = []
        for ele in range(0, 2**self.m-1):
            e = 1                               # 该元素的conjugates的数量
            store = [ele]
            while True:
                if self.pow(ele,2**e) in store:
                    break
                store.append( self.pow(ele,2**e) )
                e = e + 1
            assert type(store)==list
            if (ele not in already_record):
                already_record.extend(store)
                print('alpha^%d leading cyclotomicCoset (size: %d):  ' % (ele,e), end='')
                print(store)
        print("")

    # 其他功能：打印 所有最小多项式（基于元素的共轭对）
    def print_minimalPolynomials(self):
        already_record = []
        for ele in range(-1, 2**self.m-1):
            e = 1                               # 该元素的conjugates的数量
            store = []
            while True:
                if self.pow(ele,2**e) == ele:
                    break
                store.append( self.pow(ele,2**e) )
                e = e + 1
            store.insert(0, ele)
            assert type(store)==list
            if (ele not in already_record):
                already_record.extend(store)
                result_poly = np.array([0], dtype=np.int32)
                for index in range(0, e):
                    factors = np.array([self.addinverse( self.pow(ele,2**index) ), 0] , dtype=np.int32)
                    result_poly = self.poly_mul(result_poly, factors)
                print(f'alpha^{store} constitute minimal polynomial:  ', end="")
                print(result_poly.tolist())
        print("")

    # 其他功能：打印 对应的BCH码 ( n = 2^m-1 )，t=? 的生成多项式gx，并返回gx
    def print_BCH_gx(self, t: int):
        BCH_n = 2**self.m - 1
        if self.m < 3:
            raise NotImplementedError("[ERROR] BCH gx: m should >= 3 .")
        if t >= 2**(self.m-1):
            raise NotImplementedError("[ERROR] BCH gx: t should < 2^(m-1) .")
        process_table = list(range(1,2*t+1))
        final_result_poly = np.array([0], dtype=np.int32)
        while len(process_table) != 0:
            ele = process_table[0]
            e = 1                               # 该元素的conjugates的数量
            store = []
            while True:
                if self.pow(ele,2**e) == ele:
                    break
                store.append( self.pow(ele,2**e) )
                e = e + 1
            store.insert(0, ele)
            assert type(store)==list
            result_poly = np.array([0], dtype=np.int32)
            for index in range(0, e):
                factors = np.array([self.addinverse( self.pow(ele,2**index) ), 0] , dtype=np.int32)
                result_poly = self.poly_mul(result_poly, factors)
            process_table = np.setdiff1d(process_table, store)
            final_result_poly = self.poly_mul(final_result_poly, result_poly)
        print(f'baseGF(2) , extGF(2^{self.m})')
        print(f'm = {self.m} , t = {t}')
        checkbits = len(final_result_poly)-1
        BCH_k = BCH_n - checkbits
        print(f'BCH ( {BCH_n} , {BCH_k} ) , #checkbits = {checkbits}')
        print(f'g(X): {final_result_poly}\n')
        assert type(final_result_poly) == np.ndarray
        return final_result_poly

    def print_RS_gx(self, t: int):
        RS_n = 2**self.m**1 - 1
        if 2*t >= 2**self.m:
            raise NotImplementedError("[ERROR] RS gx: 2t should < baseGF(2^(m)) .")
        final_result_poly = np.array([0], dtype=np.int32)
        for ele in range(1, 2*t+1):
            factor = np.array([self.addinverse(ele), 0] , dtype=np.int32)
            final_result_poly = self.poly_mul(final_result_poly, factor)
        print(f'baseGF(2^{self.m}) , extGF(2^{self.m}^1)')
        print(f'm = {self.m} , t = {t} , distance = 2*{t}+1 = {2*t+1}')
        checksymbols = len(final_result_poly)-1
        RS_k = RS_n - checksymbols
        print(f'RS ( {RS_n} symbols , {RS_k} symbols ) , #checksymbols = {checksymbols}')
        print(f'RS ( {RS_n*self.m} bits, {RS_k*self.m} bits) , #checkbits = {checksymbols*self.m}  , when GF-symbol is mapped into binary form [BinaryRepresentation]')
        print(f'g(X): {final_result_poly}\n')
        assert type(final_result_poly) == np.ndarray
        return final_result_poly




########################### 以下为测试 ##########################
if __name__ == "__main__":
    polynomial = np.array([1,1,0,0,1], dtype=np.int32)
    myGF2 = GF2_map(polynomial, 4)
    print(myGF2.table_exp2tuple)
    print(myGF2.table_tupleInt2exp)

    print(myGF2.convert_exp2tuple(7))
    print(myGF2.add(3,4))
    print(myGF2.pow(1,2**1))
    print(myGF2.poly_add([-1,2,4,6,14],[1,2,3,4,-1]))
    print(myGF2.poly_addinverse([-1,2,4,6,14]))
    print(myGF2.poly_mul([-1,2,4,6,14],[1,2,3,4,-1]))
    print(myGF2.order_of_element(9))
    print(myGF2.poly_div_euclidmod([-1,2,4,6,14],[1,2,3]))
    print(myGF2.poly_div_euclidmod([-1,13,13,2,4,6,14],[1,2,13,3]))
    fx = np.array( [ 1,-1,7,12,14], dtype=np.int32)
    print( myGF2.poly_function_value(fx, 6 )  )
    myGF2.print_elements_order()
    myGF2.print_elements_conjugates()
    myGF2.print_elements_cyclotomicCoset()
    myGF2.print_minimalPolynomials()
    myGF2.print_BCH_gx(3)
    myGF2.print_RS_gx(2)

    print(myGF2.poly_function_derivative([3, 3, 3, 3, 3,3,3,3,3,3]))


    

    


