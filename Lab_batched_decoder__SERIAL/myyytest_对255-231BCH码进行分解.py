from GF2_map import GF2_map
import numpy as np

'''
if __name__ == "__main__":
    polynomial = np.array([1,1,1], dtype=np.int32)
    myGF2 = GF2_map(polynomial, 2)
    print(myGF2.table_exp2tuple)
    print(myGF2.table_tupleInt2exp)
    '''


if __name__ == "__main__":
    polynomial = np.array([1,1,0,0,0,0,1,1,1], dtype=np.int32)
    myGF2 = GF2_map(polynomial, 8)
    print(myGF2.table_exp2tuple)
    print(myGF2.table_tupleInt2exp)
    myGF2.print_elements_cyclotomicCoset()
    print(myGF2.order_of_element(3))
    myGF2.print_minimalPolynomials()






    # calculate
    ele1 = -1* np.ones([3+1], dtype=np.int32)
    ele1[0]=myGF2.addinverse(0)
    ele1[3]=0
    ele2 = np.array([myGF2.addinverse(0), 0] , dtype=np.int32)
    q , _ =myGF2.poly_div_euclidmod(ele1, ele2)
    print(q)
    
    ele3 = -1* np.ones([9+1], dtype=np.int32)
    ele3[0]=myGF2.addinverse(0)
    ele3[9]=0
    ele4 = myGF2.poly_mul(ele2, q)
    q , _ =myGF2.poly_div_euclidmod(ele3, ele4)
    print(q)



    # validate --- GF(2^6) alpha^7         is         9th primitive root of unity
    ele1 = myGF2.addinverse(myGF2.pow(7,1))
    ele2 = myGF2.addinverse(myGF2.pow(7,2))
    tmp = myGF2.poly_mul([ele1, 0] ,  [ele2, 0] )

    ele3 = myGF2.addinverse(myGF2.pow(7,4))
    tmp = myGF2.poly_mul(tmp ,  [ele3, 0] )
    ele3 = myGF2.addinverse(myGF2.pow(7,8))
    tmp = myGF2.poly_mul(tmp ,  [ele3, 0] )
    ele3 = myGF2.addinverse(myGF2.pow(7,7))
    tmp = myGF2.poly_mul(tmp ,  [ele3, 0] )
    ele3 = myGF2.addinverse(myGF2.pow(7,5))
    tmp = myGF2.poly_mul(tmp ,  [ele3, 0] )
    print(tmp)




    # 假设原始数组是一维的，将换行拼接成完整数组
    array = np.array([
        0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, 0
    ])

    # 查找值为0的索引
    zero_indices = np.where(array == 0)[0]

    print("值为0的索引位置：", zero_indices)

    myGF2.print_BCH_gx(t=3)
