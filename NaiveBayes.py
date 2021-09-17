# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:21:25 2020

@author: wangmaolin
"""

import numpy as np
import pandas as pd

def NaiveBayes(trainData, labels, features):
    labels = list(labels)    #转换为list类型
    P_y = {}       #存入label的概率
    for label in labels:
        # p = count(y) / count(Y)
        P_y[label] = labels.count(label)/float(len(labels))  
        #求label与feature同时发生的概率
    P_xy = {}
    for y in P_y.keys():
        # labels中出现y值的所有数值的下标索引
        y_index = [i for i, label in enumerate(labels) if label == y]
        # features[0] 在trainData[:,0]中出现的值的所有下标索引
        for j in range(len(features)):      
            x_index = [i for i, feature in enumerate(trainData[:,j]) if feature == features[j]]
            # set(x_index)&set(y_index)列出两个表相同的元素
            xy_count = len(set(x_index) & set(y_index))  
            pkey = str(features[j]) + '*' + str(y)
            P_xy[pkey] = xy_count / float(len(labels))

    #求条件概率
    P = {}
    for y in P_y.keys():
        for x in features:
            pkey = str(x) + '|' + str(y)
            #P[X/Y] = P[XY]/P[Y]
            P[pkey] = P_xy[str(x)+'*'+str(y)] / float(P_y[y])    

    #求所属类别
    F = {}   #属于各个类别的概率
    for y in P_y:
        F[y] = P_y[y]
        for x in features:
            #P[y/X] = P[X/y]*P[y]/P[X]，分母相等，比较分子即可，所以有F=P[X/y]*P[y]=P[x1/Y]*P[x2/Y]*P[y]
            F[y] = F[y]*P[str(x)+'|'+str(y)]     
    #概率最大值对应的类别
    features_label = max(F, key=F.get)  
    return features_label

if __name__ == '__main__':
    dataSet = pd.read_csv("./data/naivebayes.txt", sep="\t", header=None)
    dataSet.rename(columns={0:"age", 1:"sex", 2:"cp", 3:"trestbps", 4:"chol",5:"fbs",6:"restecg",7:"thalach",8:"exang",9:"oldpeak",10:"slope",11:"ca",12:"thal",13:"num"}, inplace=True)
    dataSetNP = np.array(dataSet)                         #共297行数据
    trainData = dataSetNP[0:249:,0:dataSetNP.shape[1]-1]  #将前250行作为训练集
    labels = dataSetNP[0:249:,dataSetNP.shape[1]-1]       #训练数据所对应的所属类型Y
    #输出后47行的数据预测结果
    for i in range(47):
        features = dataSetNP[250+i,0:dataSetNP.shape[1]-1]
        # 该特征应属于哪一类
        result = NaiveBayes(trainData, labels, features)
        print((int)(result))
    