# -*- coding: utf-8 -*-
print("-"*81)
print()
print("\t\t\t\t\t导入要用到的库\n\
        \t\tnumpy,pandas seaborn,sklearn.metrics.roc_auc_score\n\
        \t\tsklearn.ensemble,matplotlib,pyplot")
        
print()       
print("-"*81)
print()
###############################################
import time
import pandas as pd
import numpy as np   
import warnings 
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
time.sleep(1)
sns.set(style="white", color_codes=True)
print("-"*81)
print()
print("\t\t\t\t\t读入数据")       
print()       
print("-"*81)
print()
Train = pd.read_csv("train_r.csv") 
Test = pd.read_csv("test_r.csv")

sns.countplot(x="TARGET", data=Train)
plt.title("result")
plt.show()
#########################################################
print("-"*81)
print()
print("\t\t\t\t\t数据处理")  
print("\t\t\t\t将数据中数值型空值替换为平均值\
        \n\t\t\t\t将数据中非数值性替换为数值型")     
print()       
print("-"*81)
print()

for feat in Train.columns:
    if Train[feat].dtype == 'float64':
        Train[feat][np.isnan(Train[feat])] = Train[feat].mean()
        Test[feat][np.isnan(Test[feat])] = Test[feat].mean()
      
    elif Train[feat].dtype == 'object':
        Train[feat][Train[feat] != Train[feat]] = Train[feat].value_counts().index[0]
        Test[feat][Test[feat] != Test[feat]] = Test[feat].value_counts().index[0]
for feat in Train.columns:
    if Train[feat].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(Train[feat].values) + list(Test[feat].values)))
        Train[feat]   = lbl.transform(list(Train[feat].values))
        Test[feat]  = lbl.transform(list(Test[feat].values))  



y = Train['TARGET'].values
Train.drop('TARGET',inplace= True , axis = 1)
print("\t\t\t 数据展示")
print(Train.head())
time.sleep(3)
########################################################################
print("-"*81)
print()
print("\t\t\t\t\t利用随机森林训练模型")     
print()       
print("-"*81)
print()
time.sleep(1)

train  = Train.values
target = pd.read_csv('target.csv',header = None )
target = target.values
target = target.ravel()
test = pd.read_csv('test_r.csv')
test = test.values
clf1 = RandomForestClassifier(n_estimators=10, max_depth=1, random_state=4,min_samples_split=5,min_samples_leaf=2)

scores = cross_validation.cross_val_score(clf1, train,y,scoring='roc_auc',cv=5 )

print(clf1.fit(train,y))
time.sleep(0.5)
print("-"*81)
print("训练模型对训练集 ROC_AUC 值为："+ str(scores.mean())[:6])
print()
y_p1 = clf1.predict_proba(test)
print("-"*81)
s1 = roc_auc_score(target,y_p1[:,1])
print("训练模型对预测集 ROC_AUC 值为："+ str(s1)[:6])
time.sleep(0.5)
###############################################################
print("-"*81)
print()
print("\t\20t\t\t\t调整随机森林训练模型")       
print()       
print("-"*81)
print()
n_estimators=20
max_depth=5 
random_state=8
min_samples_split=2
min_samples_leaf=2
big =0
while big <= 10:
    n_estimators = int(input("请修改n_estimators（迭代次数）：\n"))
    max_depth = int(input("请输入修改后的max_depth(决策树深的深度)\n"))
    random_state = int(input ("请输入修改后的random_state(随机数)\n"))
    min_samples_split = int(input("请输入修改后的 min_samples_split(最小划分)\n"))
    clf2 = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,\
                                random_state= random_state,min_samples_split=min_samples_split,\
                                min_samples_leaf=min_samples_leaf)

    scores = cross_validation.cross_val_score(clf2, train,y,scoring='roc_auc',cv=5 )
    print(clf2.fit(train,y))
    print("-"*81)
    print()
    print("训练模型对训练集 ROC_AUC 值为："+ str(scores.mean())[:6])
    print()
    y_p2 = clf2.predict_proba(test)
    print("-"*81)
    print()
    s2 = roc_auc_score(target,y_p2[:,1])
    print("训练模型对预测集 ROC_AUC 值为："+ str(s2)[:6])
    print()

    print("-"*81)
    print()
    print("ROC_AUC 值提高了："+ str((s2 - s1)/s1*100)[:5]+' %')
    print()
    print("-"*81)
    time.sleep(1.5)
    big = (s2 - s1)/s1*100

print("\n\t\t 经过调参结果提升超过10%\n\n\n")
time.sleep(1)
##############################################################
print("-"*81)
print()
print("\t\t选取随机森林重要性高的训练模型")    
print()       
print("-"*81)
print()        
time.sleep(1)
feat_imp = pd.Series(clf2.feature_importances_, index=Train.columns)
feat_imp.sort_values(inplace=True)
ax = feat_imp.tail(20).plot(kind='barh', figsize=(10,7), title='Feature importance')
important = list(feat_imp[-21:].index)
plt.show()

train_i = Train[important].values
test_i = Test[important].values
clfi =RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,\
                                random_state= random_state,min_samples_split=min_samples_split,\
                                min_samples_leaf=min_samples_leaf)
scores = cross_validation.cross_val_score(clfi, train_i,y,scoring='roc_auc',cv=5 )

print(clfi.fit(train_i,y))
print("-"*81)
print()
print("训练模型对训练集 ROC_AUC 值为："+ str(scores.mean())[:6])
print()

y_pi = clfi.predict_proba(test_i)
print("-"*81)
print()
si = roc_auc_score(target,y_pi[:,1])
print("训练模型对预测集 ROC_AUC 值为："+ str(si)[:6])
print()

print("-"*81)
print()
print("ROC_AUC 值提高了："+ str((si - s2)/s2*100)[:5]+" % (" +str(si - s2)[:6] +")" )
print()
print("-"*81)