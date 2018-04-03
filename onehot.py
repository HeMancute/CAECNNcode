#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/4/3 0003 10:23
# @Author  :
# @Software: PyCharm

#one hot编码，m个可能值，转换成2元可能互斥特征，从而使得数据变得稀疏
#

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()

#fit后面四个样本，得到两个参数(实际操作中需要fit多少个元素？？)
    #enc.n_values_ 是每个样本中每一维度特征的可能数
    #enc.active_features_ 是上面可能数的累加
enc.fit([[0, 0, 9], [1, 1, 3],[1,0,8],
         [0,0,8],[0,0,4],[0,0,6],
         [0,0,5],[0,0,7],
         [0, 2, 1],[1, 0, 2]])


print ("enc.n_values_ is:",enc.n_values_)
print ("enc.feature_indices_ is:",enc.feature_indices_)

print (enc.transform([[0, 1, 7]]).toarray())