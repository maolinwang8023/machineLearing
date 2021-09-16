# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:42:55 2020

@author: wangmaolin
"""
import numpy as np
import pandas as pd
from collections import Counter

class KNNClassifier:
    def __init__(self,k):
        assert k>=1,'k must be valid'
        self.k=k
        self._x_train=None
        self._y_train=None
    def fit(self,x_train,y_train):
        self._x_train=x_train
        self._y_train=y_train
        return self
    def _predict(self,x):
        d=[np.sqrt(np.sum((x_i-x)**2)) for x_i in self._x_train]
        nearest=np.argsort(d)
        top_k=[self._y_train[i] for i in nearest[:self.k]]
        votes=Counter(top_k)
        return votes.most_common(1)[0][0]
    def predict(self,X_predict):
        y_predict=[self._predict(x1) for x1 in X_predict]
        return np.array(y_predict)
    def __repr__(self):
        return 'knn(k=%d):'%self.k
    def score(self,x_test,y_test):
        y_predict=self.predict(x_test)
        return sum(y_predict==y_test)/len(x_test)
    
# 打开数据文件
def loadDataSet(csv_path):    
    csv_file = csv_path    
    csv_data = pd.read_csv(csv_file, low_memory=False)    
    csv_df = pd.DataFrame(csv_data)    
    return csv_df

#导入数据
data = loadDataSet("processed.cleveland.data")
#拆分属性数据,前200行作为训练集
train_data_x = data.iloc[0:200,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
train_data_y = data.iloc[0:200,13]
#拆分属性数据,后97行作为测试集
test_data_x = data.iloc[200:297,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
test_data_y = data.iloc[200:297,13]
#将其转化为array
train_data_x = np.array(train_data_x)
train_data_y = np.array(train_data_y)
test_data_x = np.array(test_data_x)
test_data_y = np.array(test_data_y)

my_knn=KNNClassifier(k=3)
my_knn.fit(train_data_x,train_data_y)
predict_result = my_knn.predict(test_data_x)
print('预测结果',predict_result)
print('预测准确率',knn.score(test_data_x , test_data_y))  


