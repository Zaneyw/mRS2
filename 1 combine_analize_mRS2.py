#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import DataFrame
import os


# In[2]:


df = pd.read_csv('./dataset/3.1 stroke_mRS2.csv')
df


# In[3]:


Y = np.array(df.pop('mRS_2'))
X = np.array(df)


# In[4]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler
X = min_max_scaler().fit_transform(X)


# In[5]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3,random_state = 3)


# In[6]:


C_train = x_train[:,-4:]
R_train = x_train[:,:10]
S_train = x_train[:,10]
C_R_train = x_train[:,[11,12,13,14,0,1,2,3,4,5,6,7,8,9]]
C_S_train = x_train[:,[11,12,13,14,10]]
R_S_train = x_train[:,:11]
ALL_train = x_train[:,:]


# In[7]:


C_test = x_test[:,-4:]
R_test = x_test[:,:10]
S_test = x_test[:,10]
C_R_test = x_test[:,[11,12,13,14,0,1,2,3,4,5,6,7,8,9]]
C_S_test = x_test[:,[11,12,13,14,10]]
R_S_test = x_test[:,:11]
ALL_test = x_test[:,:]


# In[8]:


S_train = S_train.reshape(-1,1)
S_test = S_test.reshape(-1,1)


# <font color=#0099ff  size=5 face="黑体">方法1：DT</font>

# In[11]:


from sklearn import tree 
from sklearn.metrics import accuracy_score,auc,roc_curve
from sklearn import metrics
DT_classifier = tree.DecisionTreeClassifier(criterion="entropy"
                                  ,random_state=30 # 
                                  ,splitter="random")
DT_classifier.fit(C_train,y_train)
DT_train_pred = DT_classifier.predict(C_train)
DT_test_pred = DT_classifier.predict(C_test)
DT_train_acc = accuracy_score(y_train, DT_train_pred)  
DT_test_acc = accuracy_score(y_test, DT_test_pred)  
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DT_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DT_test_pred))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DT_test_pred)))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DT_test_pred)))


# In[13]:


import matplotlib.pyplot as plt
DT_test_proba = DT_classifier.predict_proba(C_test)
fpr,tpr,threshold = roc_curve(y_test, DT_test_proba[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

lw=2

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('DT_C.svg', dpi=300,format="svg")

plt.show()


# In[14]:


DT_classifier = tree.DecisionTreeClassifier(criterion="entropy"
                                  ,random_state=30 # 保证模型稳定，每一次运行，所选取的特征不变
                                  ,splitter="random")
DT_classifier.fit(R_train,y_train)
DT_train_pred = DT_classifier.predict(R_train)
DT_test_pred = DT_classifier.predict(R_test)
DT_train_acc = accuracy_score(y_train, DT_train_pred)  
DT_test_acc = accuracy_score(y_test, DT_test_pred)  
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DT_test_pred))) #预测准确率输出
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DT_test_pred))) #预测宏平均精确率输出
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DT_test_pred)))#预测宏平均召回率输出
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DT_test_pred)))#预测宏平均f1-score输出


# In[15]:


DT_test_proba = DT_classifier.predict_proba(R_test)
fpr,tpr,threshold = roc_curve(y_test, DT_test_proba[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
# print(fpr.shape, tpr.shape, threshold.shape)
#print('fpr= ', fpr)
#print('tpr= ', tpr)
#print('threshold= ',threshold)
lw=2

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('DT_R.svg', dpi=300,format="svg")

plt.show()


# In[18]:


DT_classifier = tree.DecisionTreeClassifier(criterion="entropy"
                                  ,random_state=30 # 保证模型稳定，每一次运行，所选取的特征不变
                                  ,splitter="random")
DT_classifier.fit(S_train,y_train)
DT_train_pred = DT_classifier.predict(S_train)
DT_test_pred = DT_classifier.predict(S_test)
DT_train_acc = accuracy_score(y_train, DT_train_pred)  
DT_test_acc = accuracy_score(y_test, DT_test_pred)  
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DT_test_pred))) #预测准确率输出
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DT_test_pred))) #预测宏平均精确率输出
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DT_test_pred)))#预测宏平均召回率输出
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DT_test_pred)))#预测宏平均f1-score输出

DT_test_proba = DT_classifier.predict_proba(S_test)
fpr,tpr,threshold = roc_curve(y_test, DT_test_proba[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
# print(fpr.shape, tpr.shape, threshold.shape)
#print('fpr= ', fpr)
#print('tpr= ', tpr)
#print('threshold= ',threshold)
lw=2

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('DT_S.svg', dpi=300,format="svg")

plt.show()


# In[21]:


DT_classifier = tree.DecisionTreeClassifier(criterion="entropy"
                                  ,random_state=30 # 保证模型稳定，每一次运行，所选取的特征不变
                                  ,splitter="random")
DT_classifier.fit(C_R_train,y_train)
DT_train_pred = DT_classifier.predict(C_R_train)
DT_test_pred = DT_classifier.predict(C_R_test)
DT_train_acc = accuracy_score(y_train, DT_train_pred)  
DT_test_acc = accuracy_score(y_test, DT_test_pred)  
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DT_test_pred))) #预测准确率输出
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DT_test_pred))) #预测宏平均精确率输出
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DT_test_pred)))#预测宏平均召回率输出
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DT_test_pred)))#预测宏平均f1-score输出


# In[22]:


DT_test_proba = DT_classifier.predict_proba(C_R_test)
fpr,tpr,threshold = roc_curve(y_test, DT_test_proba[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
# print(fpr.shape, tpr.shape, threshold.shape)
#print('fpr= ', fpr)
#print('tpr= ', tpr)
#print('threshold= ',threshold)
lw=2

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('DT_C_R.svg', dpi=300,format="svg")

plt.show()


# In[23]:


DT_classifier = tree.DecisionTreeClassifier(criterion="entropy"
                                  ,random_state=30 # 保证模型稳定，每一次运行，所选取的特征不变
                                  ,splitter="random")
DT_classifier.fit(C_S_train,y_train)
DT_train_pred = DT_classifier.predict(C_S_train)
DT_test_pred = DT_classifier.predict(C_S_test)
DT_train_acc = accuracy_score(y_train, DT_train_pred)  
DT_test_acc = accuracy_score(y_test, DT_test_pred)  
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DT_test_pred))) #预测准确率输出
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DT_test_pred))) #预测宏平均精确率输出
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DT_test_pred)))#预测宏平均召回率输出
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DT_test_pred)))#预测宏平均f1-score输出

DT_test_proba = DT_classifier.predict_proba(C_S_test)
fpr,tpr,threshold = roc_curve(y_test, DT_test_proba[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
# print(fpr.shape, tpr.shape, threshold.shape)
#print('fpr= ', fpr)
#print('tpr= ', tpr)
#print('threshold= ',threshold)
lw=2

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('DT_C_S.svg', dpi=300,format="svg")

plt.show()


# In[24]:


DT_classifier = tree.DecisionTreeClassifier(criterion="entropy"
                                  ,random_state=30 # 保证模型稳定，每一次运行，所选取的特征不变
                                  ,splitter="random")
DT_classifier.fit(R_S_train,y_train)
DT_train_pred = DT_classifier.predict(R_S_train)
DT_test_pred = DT_classifier.predict(R_S_test)
DT_train_acc = accuracy_score(y_train, DT_train_pred)  
DT_test_acc = accuracy_score(y_test, DT_test_pred)  
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DT_test_pred))) #预测准确率输出
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DT_test_pred))) #预测宏平均精确率输出
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DT_test_pred)))#预测宏平均召回率输出
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DT_test_pred)))#预测宏平均f1-score输出

DT_test_proba = DT_classifier.predict_proba(R_S_test)
fpr,tpr,threshold = roc_curve(y_test, DT_test_proba[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
# print(fpr.shape, tpr.shape, threshold.shape)
#print('fpr= ', fpr)
#print('tpr= ', tpr)
#print('threshold= ',threshold)
lw=2

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('DT_R_S.svg', dpi=300,format="svg")

plt.show()


# In[14]:


from sklearn import tree 
from sklearn.metrics import accuracy_score,auc,roc_curve
from sklearn import metrics
import matplotlib.pyplot as plt
DT_classifier = tree.DecisionTreeClassifier(criterion="entropy"
                                  ,random_state=6# 保证模型稳定，每一次运行，所选取的特征不变
                                  ,splitter="random")
DT_classifier.fit(ALL_train,y_train)
DT_train_pred = DT_classifier.predict(ALL_train)
DT_test_pred = DT_classifier.predict(ALL_test)
DT_train_acc = accuracy_score(y_train, DT_train_pred)  
DT_test_acc = accuracy_score(y_test, DT_test_pred)  
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DT_test_pred))) #预测准确率输出
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DT_test_pred))) #预测宏平均精确率输出
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DT_test_pred)))#预测宏平均召回率输出
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DT_test_pred)))#预测宏平均f1-score输出

DT_test_proba = DT_classifier.predict_proba(ALL_test)
fpr,tpr,threshold = roc_curve(y_test, DT_test_proba[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
# print(fpr.shape, tpr.shape, threshold.shape)
#print('fpr= ', fpr)
#print('tpr= ', tpr)
#print('threshold= ',threshold)
lw=2

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('DT_ALL.svg', dpi=300,format="svg")

plt.show()


# <font color=#0099ff  size=5 face="黑体">方法2：SVM 支持向量机</font>

# In[62]:


from sklearn import svm
SVM_classifier = svm.SVC(C=1, kernel='linear',probability=True)
SVM_classifier.fit(C_train, y_train)

SVM_train_pred = SVM_classifier.predict(C_train)
SVM_test_pred = SVM_classifier.predict(C_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, SVM_test_pred))) #预测准确率输出
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test,  SVM_test_pred))) #预测宏平均精确率输出
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test,  SVM_test_pred)))#预测宏平均召回率输出
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test,  SVM_test_pred)))#预测宏平均f1-score输出

SVM_test_proba = SVM_classifier.predict_proba(C_test)

fpr,tpr,threshold = roc_curve(y_test, SVM_test_proba[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
# print(fpr.shape, tpr.shape, threshold.shape)
#print('fpr= ', fpr)
#print('tpr= ', tpr)
#print('threshold= ',threshold)


plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('SVM_C.svg', dpi=300,format="svg")

plt.show()


# In[38]:


SVM_classifier = svm.SVC(C=2, kernel='linear',probability=True)

SVM_classifier.fit(R_train, y_train)               #不同

SVM_train_pred = SVM_classifier.predict(R_train)   #不同
SVM_test_pred = SVM_classifier.predict(R_test)     #不同

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, SVM_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test,  SVM_test_pred))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test,  SVM_test_pred)))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test,  SVM_test_pred)))

SVM_test_proba = SVM_classifier.predict_proba(R_test)

fpr,tpr,threshold = roc_curve(y_test, SVM_test_proba[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('SVM_R.svg', dpi=300,format="svg")   #不同

plt.show()


# In[37]:


SVM_classifier = svm.SVC(C=2, kernel='linear',probability=True)

SVM_classifier.fit(S_train, y_train)               #不同

SVM_train_pred = SVM_classifier.predict(S_train)   #不同
SVM_test_pred = SVM_classifier.predict(S_test)     #不同

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, SVM_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test,  SVM_test_pred))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test,  SVM_test_pred)))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test,  SVM_test_pred)))

SVM_test_proba = SVM_classifier.predict_proba(S_test)   #不同

fpr,tpr,threshold = roc_curve(y_test, SVM_test_proba[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('SVM_S.svg', dpi=300,format="svg")   #不同

plt.show()


# In[39]:


SVM_classifier = svm.SVC(C=1, kernel='linear',probability=True)

SVM_classifier.fit(C_R_train, y_train)               #不同

SVM_train_pred = SVM_classifier.predict(C_R_train)   #不同
SVM_test_pred = SVM_classifier.predict(C_R_test)     #不同

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, SVM_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test,  SVM_test_pred))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test,  SVM_test_pred)))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test,  SVM_test_pred)))

SVM_test_proba = SVM_classifier.predict_proba(C_R_test)   #不同

fpr,tpr,threshold = roc_curve(y_test, SVM_test_proba[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('SVM_C_R.svg', dpi=300,format="svg")   #不同

plt.show()


# In[40]:


SVM_classifier = svm.SVC(C=1, kernel='linear',probability=True)

SVM_classifier.fit(C_S_train, y_train)               #不同

SVM_train_pred = SVM_classifier.predict(C_S_train)   #不同
SVM_test_pred = SVM_classifier.predict(C_S_test)     #不同

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, SVM_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test,  SVM_test_pred))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test,  SVM_test_pred)))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test,  SVM_test_pred)))

SVM_test_proba = SVM_classifier.predict_proba(C_S_test)   #不同

fpr,tpr,threshold = roc_curve(y_test, SVM_test_proba[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('SVM_C_S.svg', dpi=300,format="svg")   #不同

plt.show()


# In[41]:


SVM_classifier = svm.SVC(C=1, kernel='linear',probability=True)

SVM_classifier.fit(R_S_train, y_train)               #不同

SVM_train_pred = SVM_classifier.predict(R_S_train)   #不同
SVM_test_pred = SVM_classifier.predict(R_S_test)     #不同

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, SVM_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test,  SVM_test_pred))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test,  SVM_test_pred)))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test,  SVM_test_pred)))

SVM_test_proba = SVM_classifier.predict_proba(R_S_test)   #不同

fpr,tpr,threshold = roc_curve(y_test, SVM_test_proba[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('SVM_R_S.svg', dpi=300,format="svg")   #不同

plt.show()


# In[24]:


from sklearn import svm
SVM_classifier = svm.SVC(C=1, kernel='rbf',probability=True)

SVM_classifier.fit(ALL_train, y_train)               #不同

SVM_train_pred = SVM_classifier.predict(ALL_train)   #不同
SVM_test_pred = SVM_classifier.predict(ALL_test)     #不同

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, SVM_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test,  SVM_test_pred))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test,  SVM_test_pred)))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test,  SVM_test_pred)))

SVM_test_proba = SVM_classifier.predict_proba(ALL_test)   #不同

fpr,tpr,threshold = roc_curve(y_test, SVM_test_proba[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('SVM_ALL.svg', dpi=300,format="svg")   #不同

plt.show()


# In[ ]:





# <font color=#0099ff  size=5 face="黑体">方法3：RF随机森林</font>

# In[60]:


from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier(n_estimators = 1, criterion="gini",random_state=3)

RF_classifier.fit(C_train, y_train)                        #change
RF_train_pred = RF_classifier.predict(C_train)              #change
RF_test_pred = RF_classifier.predict(C_test)                #change

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, RF_test_pred)))
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test,  RF_test_pred)))
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test,  RF_test_pred)))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test,  RF_test_pred)))

RF_test_proba = RF_classifier.predict_proba(C_test)        #change

fpr,tpr,threshold = roc_curve(y_test, RF_test_proba[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('RF_C.svg', dpi=300,format="svg")       #change

plt.show()


# In[45]:


RF_classifier = RandomForestClassifier(n_estimators = 5, criterion="gini",random_state=2)

RF_classifier.fit(R_train, y_train)                        #change
RF_train_pred = RF_classifier.predict(R_train)              #change
RF_test_pred = RF_classifier.predict(R_test)                #change

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, RF_test_pred)))
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test,  RF_test_pred)))
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test,  RF_test_pred)))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test,  RF_test_pred)))

RF_test_proba = RF_classifier.predict_proba(R_test)        #change

fpr,tpr,threshold = roc_curve(y_test, RF_test_proba[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('RF_R.svg', dpi=300,format="svg")       #change

plt.show()


# In[46]:


RF_classifier = RandomForestClassifier(n_estimators = 5, criterion="gini",random_state=2)

RF_classifier.fit(S_train, y_train)                        #change
RF_train_pred = RF_classifier.predict(S_train)              #change
RF_test_pred = RF_classifier.predict(S_test)                #change

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, RF_test_pred)))
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test,  RF_test_pred)))
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test,  RF_test_pred)))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test,  RF_test_pred)))

RF_test_proba = RF_classifier.predict_proba(S_test)        #change

fpr,tpr,threshold = roc_curve(y_test, RF_test_proba[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('RF_S.svg', dpi=300,format="svg")       #change

plt.show()


# In[47]:


RF_classifier = RandomForestClassifier(n_estimators = 5, criterion="gini",random_state=2)

RF_classifier.fit(C_R_train, y_train)                        #change
RF_train_pred = RF_classifier.predict(C_R_train)              #change
RF_test_pred = RF_classifier.predict(C_R_test)                #change

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, RF_test_pred)))
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test,  RF_test_pred)))
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test,  RF_test_pred)))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test,  RF_test_pred)))

RF_test_proba = RF_classifier.predict_proba(C_R_test)        #change

fpr,tpr,threshold = roc_curve(y_test, RF_test_proba[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('RF_C_R.svg', dpi=300,format="svg")       #change

plt.show()


# In[57]:


RF_classifier = RandomForestClassifier(n_estimators = 2, criterion="gini",random_state=1)

RF_classifier.fit(C_S_train, y_train)                        #change
RF_train_pred = RF_classifier.predict(C_S_train)              #change
RF_test_pred = RF_classifier.predict(C_S_test)                #change

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, RF_test_pred)))
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test,  RF_test_pred)))
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test,  RF_test_pred)))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test,  RF_test_pred)))

RF_test_proba = RF_classifier.predict_proba(C_S_test)        #change

fpr,tpr,threshold = roc_curve(y_test, RF_test_proba[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('RF_C_S.svg', dpi=300,format="svg")       #change

plt.show()


# In[49]:


RF_classifier = RandomForestClassifier(n_estimators = 5, criterion="gini",random_state=2)

RF_classifier.fit(R_S_train, y_train)                        #change
RF_train_pred = RF_classifier.predict(R_S_train)              #change
RF_test_pred = RF_classifier.predict(R_S_test)                #change

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, RF_test_pred)))
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test,  RF_test_pred)))
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test,  RF_test_pred)))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test,  RF_test_pred)))

RF_test_proba = RF_classifier.predict_proba(R_S_test)        #change

fpr,tpr,threshold = roc_curve(y_test, RF_test_proba[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('RF_R_S.svg', dpi=300,format="svg")       #change

plt.show()


# In[26]:


from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier(n_estimators = 5, criterion="gini",random_state=3)

RF_classifier.fit(ALL_train, y_train)                        #change
RF_train_pred = RF_classifier.predict(ALL_train)              #change
RF_test_pred = RF_classifier.predict(ALL_test)                #change

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, RF_test_pred)))
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test,  RF_test_pred)))
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test,  RF_test_pred)))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test,  RF_test_pred)))

RF_test_proba = RF_classifier.predict_proba(ALL_test)        #change

fpr,tpr,threshold = roc_curve(y_test, RF_test_proba[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('RF_ALL.svg', dpi=300,format="svg")       #change

plt.show()


# In[ ]:





# <font color=#0099ff  size=5 face="黑体">方法4：BiRNN</font>

# In[84]:


import tensorflow.keras as K
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,LSTM,GRU,BatchNormalization
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import Dropout

model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 4)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_train.reshape(C_train.shape[0],1,C_train.shape[1])                                            #change
x_test_lstm = C_test.reshape(C_test.shape[0],1,C_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=100, verbose=1)


# In[85]:


lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('RNN_C.svg', dpi=300,format="svg")

plt.show()


# In[86]:


import tensorflow.keras as K
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,LSTM,GRU,BatchNormalization
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import Dropout

model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 10)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = R_train.reshape(R_train.shape[0],1,R_train.shape[1])                                            #change
x_test_lstm = R_test.reshape(R_test.shape[0],1,R_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=200, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('RNN_R.svg', dpi=300,format="svg")

plt.show()


# In[87]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 1)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = S_train.reshape(S_train.shape[0],1,1)                                            #change
x_test_lstm = S_test.reshape(S_test.shape[0],1,1)                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=200, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('RNN_S.svg', dpi=300,format="svg")

plt.show()


# In[88]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 14)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_R_train.reshape(C_R_train.shape[0],1,C_R_train.shape[1])                                            #change
x_test_lstm = C_R_test.reshape(C_R_test.shape[0],1,C_R_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=200, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('RNN_C_R.svg', dpi=300,format="svg")

plt.show()


# In[89]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 5)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_S_train.reshape(C_S_train.shape[0],1,C_S_train.shape[1])                                            #change
x_test_lstm = C_S_test.reshape(C_S_test.shape[0],1,C_S_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=200, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('RNN_C_S.svg', dpi=300,format="svg")

plt.show()


# In[90]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 11)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = R_S_train.reshape(R_S_train.shape[0],1,R_S_train.shape[1])                                            #change
x_test_lstm = R_S_test.reshape(R_S_test.shape[0],1,R_S_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=200, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('RNN_R_S.svg', dpi=300,format="svg")

plt.show()


# In[28]:


import tensorflow.keras as K
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,LSTM,GRU,BatchNormalization
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import Dropout
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 15)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = ALL_train.reshape(ALL_train.shape[0],1,ALL_train.shape[1])                                            #change
x_test_lstm = ALL_test.reshape(ALL_test.shape[0],1,ALL_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=100, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('RNN_ALL.svg', dpi=300,format="svg")

plt.show()


# <font color=#0099ff  size=5 face="黑体">方法5：BiLSTM</font>

# In[97]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 4)))                                                           #change
model.add(Bidirectional(LSTM(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_train.reshape(C_train.shape[0],1,C_train.shape[1])                                            #change
x_test_lstm = C_test.reshape(C_test.shape[0],1,C_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=100, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('LSTM_C.svg', dpi=300,format="svg")

plt.show()


# In[96]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 10)))                                                           #change
model.add(Bidirectional(LSTM(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = R_train.reshape(R_train.shape[0],1,R_train.shape[1])                                            #change
x_test_lstm = R_test.reshape(R_test.shape[0],1,R_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=500, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('LSTM_R.svg', dpi=300,format="svg")                                                                      #change

plt.show()


# In[98]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 1)))                                                           #change
model.add(Bidirectional(LSTM(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = S_train.reshape(S_train.shape[0],1,1)                                            #change
x_test_lstm = S_test.reshape(S_test.shape[0],1,1)                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=500, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('LSTM_S.svg', dpi=300,format="svg")                                                                      #change

plt.show()


# In[103]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 14)))                                                           #change
model.add(Bidirectional(LSTM(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_R_train.reshape(C_R_train.shape[0],1,C_R_train.shape[1])                                            #change
x_test_lstm = C_R_test.reshape(C_R_test.shape[0],1,C_R_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=100, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('LSTM_C_R.svg', dpi=300,format="svg")

plt.show()


# In[100]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 5)))                                                           #change
model.add(Bidirectional(LSTM(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_S_train.reshape(C_S_train.shape[0],1,C_S_train.shape[1])                                            #change
x_test_lstm = C_S_test.reshape(C_S_test.shape[0],1,C_S_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=500, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('LSTM_C_S.svg', dpi=300,format="svg")

plt.show()


# In[101]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 11)))                                                           #change
model.add(Bidirectional(LSTM(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = R_S_train.reshape(R_S_train.shape[0],1,R_S_train.shape[1])                                            #change
x_test_lstm = R_S_test.reshape(R_S_test.shape[0],1,R_S_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=500, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('LSTM_R_S.svg', dpi=300,format="svg")

plt.show()


# In[29]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 15)))                                                           #change
model.add(Bidirectional(LSTM(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = ALL_train.reshape(ALL_train.shape[0],1,ALL_train.shape[1])                                            #change
x_test_lstm = ALL_test.reshape(ALL_test.shape[0],1,ALL_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=100, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('LSTM_ALL.svg', dpi=300,format="svg")

plt.show()


# In[ ]:





# <font color=#0099ff  size=5 face="黑体">方法6：BiGRU</font>

# In[111]:


model = Sequential()
model.add(Bidirectional(GRU(10, return_sequences=True),
                             input_shape=(1, 4)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_train.reshape(C_train.shape[0],1,C_train.shape[1])                                            #change
x_test_lstm = C_test.reshape(C_test.shape[0],1,C_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=100, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('GRU_C.svg', dpi=300,format="svg")

plt.show()


# In[105]:


model = Sequential()
model.add(Bidirectional(GRU(10, return_sequences=True),
                             input_shape=(1, 10)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = R_train.reshape(R_train.shape[0],1,R_train.shape[1])                                            #change
x_test_lstm = R_test.reshape(R_test.shape[0],1,R_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=500, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('GRU_R.svg', dpi=300,format="svg")                                                                      #change

plt.show()


# In[106]:


model = Sequential()
model.add(Bidirectional(GRU(10, return_sequences=True),
                             input_shape=(1, 1)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = S_train.reshape(S_train.shape[0],1,1)                                            #change
x_test_lstm = S_test.reshape(S_test.shape[0],1,1)                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=500, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('GRU_S.svg', dpi=300,format="svg")                                                                      #change

plt.show()


# In[114]:


model = Sequential()
model.add(Bidirectional(GRU(10, return_sequences=True),
                             input_shape=(1, 14)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_R_train.reshape(C_R_train.shape[0],1,C_R_train.shape[1])                                            #change
x_test_lstm = C_R_test.reshape(C_R_test.shape[0],1,C_R_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=100, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('GRU_C_R.svg', dpi=300,format="svg")

plt.show()


# In[115]:


model = Sequential()
model.add(Bidirectional(GRU(10, return_sequences=True),
                             input_shape=(1, 5)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_S_train.reshape(C_S_train.shape[0],1,C_S_train.shape[1])                                            #change
x_test_lstm = C_S_test.reshape(C_S_test.shape[0],1,C_S_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=100, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('GRU_C_S.svg', dpi=300,format="svg")

plt.show()


# In[109]:


model = Sequential()
model.add(Bidirectional(GRU(10, return_sequences=True),
                             input_shape=(1, 11)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = R_S_train.reshape(R_S_train.shape[0],1,R_S_train.shape[1])                                            #change
x_test_lstm = R_S_test.reshape(R_S_test.shape[0],1,R_S_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=500, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('GRU_R_S.svg', dpi=300,format="svg")

plt.show()


# In[30]:


model = Sequential()
model.add(Bidirectional(GRU(10, return_sequences=True),
                             input_shape=(1, 15)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = ALL_train.reshape(ALL_train.shape[0],1,ALL_train.shape[1])                                            #change
x_test_lstm = ALL_test.reshape(ALL_test.shape[0],1,ALL_test.shape[1])                                                #change

history = model.fit(x_train_lstm,y_train, validation_data=(x_test_lstm, y_test), epochs=100, verbose=1)

lstm_test_proda = model.predict(x_test_lstm)
lstm_test_pred = np.int64(lstm_test_proda>0.5)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, lstm_test_pred))) 

print('混淆矩阵输出:\n',metrics.confusion_matrix(y_test,  lstm_test_pred))

print('分类报告:\n', metrics.classification_report(y_test, lstm_test_pred,digits=4))

fpr,tpr,threshold = roc_curve(y_test, lstm_test_proda) 
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}

plt.legend(loc="lower right",prop=font1)

plt.savefig('GRU_ALL.svg', dpi=300,format="svg")

plt.show()


# In[ ]:





# <font color=#0099ff  size=5 face="黑体">方法7：DA</font>

# In[119]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

DA = LinearDiscriminantAnalysis()
DA.fit(C_train,y_train)
DA_test_pred = DA.predict(C_test)
DA_test_proba = DA.predict_proba(C_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DA_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DA_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DA_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DA_test_pred)))


# In[121]:


fpr,tpr,threshold = roc_curve(y_test, DA_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('DA_C.svg', dpi=300,format="svg")
plt.show()


# In[122]:


DA = LinearDiscriminantAnalysis()
DA.fit(R_train,y_train)
DA_test_pred = DA.predict(R_test)
DA_test_proba = DA.predict_proba(R_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DA_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DA_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DA_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DA_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, DA_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('DA_R.svg', dpi=300,format="svg")
plt.show()


# In[123]:


DA = LinearDiscriminantAnalysis()
DA.fit(S_train,y_train)
DA_test_pred = DA.predict(S_test)
DA_test_proba = DA.predict_proba(S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DA_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DA_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DA_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DA_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, DA_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('DA_S.svg', dpi=300,format="svg")
plt.show()


# In[124]:


DA = LinearDiscriminantAnalysis()
DA.fit(C_R_train,y_train)
DA_test_pred = DA.predict(C_R_test)
DA_test_proba = DA.predict_proba(C_R_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DA_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DA_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DA_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DA_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, DA_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('DA_C_R.svg', dpi=300,format="svg")
plt.show()


# In[126]:


DA = LinearDiscriminantAnalysis()
DA.fit(C_S_train,y_train)
DA_test_pred = DA.predict(C_S_test)
DA_test_proba = DA.predict_proba(C_S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DA_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DA_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DA_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DA_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, DA_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('DA_C_S.svg', dpi=300,format="svg")
plt.show()


# In[127]:


DA = LinearDiscriminantAnalysis()
DA.fit(R_S_train,y_train)
DA_test_pred = DA.predict(R_S_test)
DA_test_proba = DA.predict_proba(R_S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DA_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DA_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DA_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DA_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, DA_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('DA_R_S.svg', dpi=300,format="svg")
plt.show()


# In[34]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
DA = LinearDiscriminantAnalysis()
DA.fit(ALL_train,y_train)
DA_test_pred = DA.predict(ALL_test)
DA_test_proba = DA.predict_proba(ALL_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DA_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DA_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DA_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DA_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, DA_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('DA_ALL.svg', dpi=300,format="svg")
plt.show()


# <font color=#0099ff  size=5 face="黑体">方法8：NB</font>

# In[169]:


from sklearn.naive_bayes import GaussianNB
NB = GaussianNB(var_smoothing=1e-02)
NB.fit(C_train,y_train)
NB_test_pred = NB.predict(C_test)
NB_test_proba = NB.predict_proba(C_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, NB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, NB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, NB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, NB_test_pred)))


# In[170]:


fpr,tpr,threshold = roc_curve(y_test, NB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('NB_C.svg', dpi=300,format="svg")
plt.show()


# In[175]:


NB = GaussianNB()
NB.fit(R_train,y_train)
NB_test_pred = NB.predict(R_test)
NB_test_proba = NB.predict_proba(R_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, NB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, NB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, NB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, NB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, NB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('NB_R.svg', dpi=300,format="svg")
plt.show()


# In[133]:


NB = GaussianNB()
NB.fit(S_train,y_train)
NB_test_pred = NB.predict(S_test)
NB_test_proba = NB.predict_proba(S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, NB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, NB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, NB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, NB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, NB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('NB_S.svg', dpi=300,format="svg")
plt.show()


# In[134]:


NB = GaussianNB()
NB.fit(C_R_train,y_train)
NB_test_pred = NB.predict(C_R_test)
NB_test_proba = NB.predict_proba(C_R_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, NB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, NB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, NB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, NB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, NB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('NB_C_R.svg', dpi=300,format="svg")
plt.show()


# In[135]:


NB = GaussianNB()
NB.fit(C_S_train,y_train)
NB_test_pred = NB.predict(C_S_test)
NB_test_proba = NB.predict_proba(C_S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, NB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, NB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, NB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, NB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, NB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('NB_C_S.svg', dpi=300,format="svg")
plt.show()


# In[137]:


NB = GaussianNB()
NB.fit(R_S_train,y_train)
NB_test_pred = NB.predict(R_S_test)
NB_test_proba = NB.predict_proba(R_S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, NB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, NB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, NB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, NB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, NB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('NB_R_S.svg', dpi=300,format="svg")
plt.show()


# In[35]:


from sklearn.naive_bayes import GaussianNB

NB = GaussianNB(var_smoothing=1e-09)
NB.fit(ALL_train,y_train)
NB_test_pred = NB.predict(ALL_test)
NB_test_proba = NB.predict_proba(ALL_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, NB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, NB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, NB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, NB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, NB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('NB_ALL.svg', dpi=300,format="svg")
plt.show()


# <font color=#0099ff  size=5 face="黑体">方法8：KNN</font>

# In[171]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(C_train,y_train)
knn_test_pred = knn.predict(C_test)
knn_test_proba = knn.predict_proba(C_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, knn_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, knn_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, knn_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, knn_test_pred)))


# In[172]:


fpr,tpr,threshold = roc_curve(y_test, knn_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('KNN_C.svg', dpi=300,format="svg")
plt.show()


# In[177]:


knn = KNeighborsClassifier()
knn.fit(R_train,y_train)
knn_test_pred = knn.predict(R_test)
knn_test_proba = knn.predict_proba(R_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, knn_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, knn_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, knn_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, knn_test_pred)))
fpr,tpr,threshold = roc_curve(y_test, knn_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('KNN_R.svg', dpi=300,format="svg")
plt.show()


# In[178]:


knn = KNeighborsClassifier()
knn.fit(S_train,y_train)
knn_test_pred = knn.predict(S_test)
knn_test_proba = knn.predict_proba(S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, knn_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, knn_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, knn_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, knn_test_pred)))
fpr,tpr,threshold = roc_curve(y_test, knn_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('KNN_S.svg', dpi=300,format="svg")
plt.show()


# In[179]:


knn = KNeighborsClassifier()
knn.fit(C_R_train,y_train)
knn_test_pred = knn.predict(C_R_test)
knn_test_proba = knn.predict_proba(C_R_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, knn_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, knn_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, knn_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, knn_test_pred)))
fpr,tpr,threshold = roc_curve(y_test, knn_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('KNN_C_R.svg', dpi=300,format="svg")
plt.show()


# In[180]:


knn = KNeighborsClassifier()
knn.fit(C_S_train,y_train)
knn_test_pred = knn.predict(C_S_test)
knn_test_proba = knn.predict_proba(C_S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, knn_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, knn_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, knn_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, knn_test_pred)))
fpr,tpr,threshold = roc_curve(y_test, knn_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('KNN_C_S.svg', dpi=300,format="svg")
plt.show()


# In[181]:


knn = KNeighborsClassifier()
knn.fit(R_S_train,y_train)
knn_test_pred = knn.predict(R_S_test)
knn_test_proba = knn.predict_proba(R_S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, knn_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, knn_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, knn_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, knn_test_pred)))
fpr,tpr,threshold = roc_curve(y_test, knn_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('KNN_R_S.svg', dpi=300,format="svg")
plt.show()


# In[182]:


knn = KNeighborsClassifier()
knn.fit(ALL_train,y_train)
knn_test_pred = knn.predict(ALL_test)
knn_test_proba = knn.predict_proba(ALL_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, knn_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, knn_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, knn_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, knn_test_pred)))
fpr,tpr,threshold = roc_curve(y_test, knn_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('KNN_ALL.svg', dpi=300,format="svg")
plt.show()


# <font color=#0099ff  size=5 face="黑体">方法9：MLP</font>

# In[191]:


from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(hidden_layer_sizes=(100,20),random_state=1)
MLP.fit(C_train,y_train)
MLP_test_pred = MLP.predict(C_test)
MLP_test_proba = MLP.predict_proba(C_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, MLP_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, MLP_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, MLP_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, MLP_test_pred)))


# In[192]:


fpr,tpr,threshold = roc_curve(y_test, MLP_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('MLP_C.svg', dpi=300,format="svg")
plt.show()


# In[186]:


from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(hidden_layer_sizes=(400,200),random_state=1)
MLP.fit(R_train,y_train)
MLP_test_pred = MLP.predict(R_test)
MLP_test_proba = MLP.predict_proba(R_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, MLP_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, MLP_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, MLP_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, MLP_test_pred)))
fpr,tpr,threshold = roc_curve(y_test, MLP_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('MLP_R.svg', dpi=300,format="svg")
plt.show()


# In[187]:


MLP = MLPClassifier(hidden_layer_sizes=(400,200),random_state=1)
MLP.fit(S_train,y_train)
MLP_test_pred = MLP.predict(S_test)
MLP_test_proba = MLP.predict_proba(S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, MLP_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, MLP_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, MLP_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, MLP_test_pred)))
fpr,tpr,threshold = roc_curve(y_test, MLP_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('MLP_S.svg', dpi=300,format="svg")
plt.show()


# In[188]:


MLP = MLPClassifier(hidden_layer_sizes=(400,200),random_state=1)
MLP.fit(C_R_train,y_train)
MLP_test_pred = MLP.predict(C_R_test)
MLP_test_proba = MLP.predict_proba(C_R_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, MLP_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, MLP_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, MLP_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, MLP_test_pred)))
fpr,tpr,threshold = roc_curve(y_test, MLP_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('MLP_C_R.svg', dpi=300,format="svg")
plt.show()


# In[208]:


MLP = MLPClassifier(hidden_layer_sizes=(400,200),random_state=1)
MLP.fit(C_S_train,y_train)
MLP_test_pred = MLP.predict(C_S_test)
MLP_test_proba = MLP.predict_proba(C_S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, MLP_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, MLP_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, MLP_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, MLP_test_pred)))
fpr,tpr,threshold = roc_curve(y_test, MLP_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('MLP_C_S.svg', dpi=300,format="svg")
plt.show()


# In[201]:


MLP = MLPClassifier(hidden_layer_sizes=(400,300),random_state=1)
MLP.fit(R_S_train,y_train)
MLP_test_pred = MLP.predict(R_S_test)
MLP_test_proba = MLP.predict_proba(R_S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, MLP_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, MLP_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, MLP_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, MLP_test_pred)))
fpr,tpr,threshold = roc_curve(y_test, MLP_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('MLP_R_S.svg', dpi=300,format="svg")
plt.show()


# In[43]:


from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(hidden_layer_sizes=(100,100),random_state=4)
MLP.fit(ALL_train,y_train)
MLP_test_pred = MLP.predict(ALL_test)
MLP_test_proba = MLP.predict_proba(ALL_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, MLP_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, MLP_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, MLP_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, MLP_test_pred)))
fpr,tpr,threshold = roc_curve(y_test, MLP_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('MLP_ALL.svg', dpi=300,format="svg")
plt.show()


# <font color=#0099ff  size=10 face="黑体">方法10：BAGGING</font>

# In[211]:


from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier()
bag.fit(C_train,y_train)
bag_test_pred = bag.predict(C_test)
bag_test_proba = bag.predict_proba(C_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, bag_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, bag_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, bag_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, bag_test_pred)))


# In[212]:


fpr,tpr,threshold = roc_curve(y_test, bag_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('BAG_C.svg', dpi=300,format="svg")
plt.show()


# In[213]:


bag = BaggingClassifier()
bag.fit(R_train,y_train)
bag_test_pred = bag.predict(R_test)
bag_test_proba = bag.predict_proba(R_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, bag_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, bag_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, bag_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, bag_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, bag_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('BAG_R.svg', dpi=300,format="svg")
plt.show()


# In[214]:


bag = BaggingClassifier()
bag.fit(S_train,y_train)
bag_test_pred = bag.predict(S_test)
bag_test_proba = bag.predict_proba(S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, bag_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, bag_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, bag_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, bag_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, bag_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('BAG_S.svg', dpi=300,format="svg")
plt.show()


# In[215]:


bag = BaggingClassifier()
bag.fit(C_R_train,y_train)
bag_test_pred = bag.predict(C_R_test)
bag_test_proba = bag.predict_proba(C_R_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, bag_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, bag_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, bag_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, bag_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, bag_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('BAG_C_R.svg', dpi=300,format="svg")
plt.show()


# In[216]:


bag = BaggingClassifier()
bag.fit(C_S_train,y_train)
bag_test_pred = bag.predict(C_S_test)
bag_test_proba = bag.predict_proba(C_S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, bag_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, bag_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, bag_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, bag_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, bag_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('BAG_C_S.svg', dpi=300,format="svg")
plt.show()


# In[217]:


bag = BaggingClassifier()
bag.fit(R_S_train,y_train)
bag_test_pred = bag.predict(R_S_test)
bag_test_proba = bag.predict_proba(R_S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, bag_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, bag_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, bag_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, bag_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, bag_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('BAG_R_S.svg', dpi=300,format="svg")
plt.show()


# In[46]:


from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(random_state=1)
bag.fit(ALL_train,y_train)
bag_test_pred = bag.predict(ALL_test)
bag_test_proba = bag.predict_proba(ALL_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, bag_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, bag_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, bag_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, bag_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, bag_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('BAG_ALL.svg', dpi=300,format="svg")
plt.show()


# <font color=#0099ff  size=10 face="黑体">方法11：ADBoostING</font>

# In[219]:


from sklearn.ensemble import AdaBoostClassifier
ADB = AdaBoostClassifier()
ADB.fit(C_train,y_train)
ADB_test_pred = ADB.predict(C_test)
ADB_test_proba = ADB.predict_proba(C_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, ADB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, ADB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, ADB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, ADB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, ADB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('ADB_C.svg', dpi=300,format="svg")
plt.show()


# In[220]:


ADB = AdaBoostClassifier()
ADB.fit(R_train,y_train)
ADB_test_pred = ADB.predict(R_test)
ADB_test_proba = ADB.predict_proba(R_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, ADB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, ADB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, ADB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, ADB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, ADB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('ADB_R.svg', dpi=300,format="svg")
plt.show()


# In[221]:


ADB = AdaBoostClassifier()
ADB.fit(S_train,y_train)
ADB_test_pred = ADB.predict(S_test)
ADB_test_proba = ADB.predict_proba(S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, ADB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, ADB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, ADB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, ADB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, ADB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('ADB_S.svg', dpi=300,format="svg")
plt.show()


# In[222]:


ADB = AdaBoostClassifier()
ADB.fit(C_R_train,y_train)
ADB_test_pred = ADB.predict(C_R_test)
ADB_test_proba = ADB.predict_proba(C_R_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, ADB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, ADB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, ADB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, ADB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, ADB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('ADB_C_R.svg', dpi=300,format="svg")
plt.show()


# In[223]:


ADB = AdaBoostClassifier()
ADB.fit(C_S_train,y_train)
ADB_test_pred = ADB.predict(C_S_test)
ADB_test_proba = ADB.predict_proba(C_S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, ADB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, ADB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, ADB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, ADB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, ADB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('ADB_C_S.svg', dpi=300,format="svg")
plt.show()


# In[224]:


ADB = AdaBoostClassifier()
ADB.fit(R_S_train,y_train)
ADB_test_pred = ADB.predict(R_S_test)
ADB_test_proba = ADB.predict_proba(R_S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, ADB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, ADB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, ADB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, ADB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, ADB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('ADB_R_S.svg', dpi=300,format="svg")
plt.show()


# In[50]:


from sklearn.ensemble import AdaBoostClassifier
ADB = AdaBoostClassifier(random_state=4)
ADB.fit(ALL_train,y_train)
ADB_test_pred = ADB.predict(ALL_test)
ADB_test_proba = ADB.predict_proba(ALL_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, ADB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, ADB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, ADB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, ADB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, ADB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('ADB_ALL.svg', dpi=300,format="svg")
plt.show()


# <font color=#0099ff  size=10 face="黑体">方法11：XGBoostING</font>

# In[226]:


from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(C_train,y_train)
XGB_test_pred = XGB.predict(C_test)
XGB_test_proba = XGB.predict_proba(C_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, XGB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, XGB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, XGB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, XGB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, XGB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('XGB_C.svg', dpi=300,format="svg")
plt.show()


# In[227]:


from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(R_train,y_train)
XGB_test_pred = XGB.predict(R_test)
XGB_test_proba = XGB.predict_proba(R_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, XGB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, XGB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, XGB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, XGB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, XGB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('XGB_R.svg', dpi=300,format="svg")
plt.show()


# In[228]:


from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(S_train,y_train)
XGB_test_pred = XGB.predict(S_test)
XGB_test_proba = XGB.predict_proba(S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, XGB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, XGB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, XGB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, XGB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, XGB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('XGB_S.svg', dpi=300,format="svg")
plt.show()


# In[229]:


from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(C_R_train,y_train)
XGB_test_pred = XGB.predict(C_R_test)
XGB_test_proba = XGB.predict_proba(C_R_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, XGB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, XGB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, XGB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, XGB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, XGB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('XGB_C_R.svg', dpi=300,format="svg")
plt.show()


# In[230]:


from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(C_S_train,y_train)
XGB_test_pred = XGB.predict(C_S_test)
XGB_test_proba = XGB.predict_proba(C_S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, XGB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, XGB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, XGB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, XGB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, XGB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('XGB_C_S.svg', dpi=300,format="svg")
plt.show()


# In[231]:


from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(R_S_train,y_train)
XGB_test_pred = XGB.predict(R_S_test)
XGB_test_proba = XGB.predict_proba(R_S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, XGB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, XGB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, XGB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, XGB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, XGB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('XGB_R_S.svg', dpi=300,format="svg")
plt.show()


# In[232]:


from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(ALL_train,y_train)
XGB_test_pred = XGB.predict(ALL_test)
XGB_test_proba = XGB.predict_proba(ALL_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, XGB_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, XGB_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, XGB_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, XGB_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, XGB_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('XGB_ALL.svg', dpi=300,format="svg")
plt.show()


# <font color=#0099ff  size=10 face="黑体">方法12：GBDT</font>

# In[233]:


from sklearn.ensemble import GradientBoostingClassifier
GBDT = GradientBoostingClassifier()
GBDT.fit(C_train,y_train)
GBDT_test_pred = GBDT.predict(C_test)
GBDT_test_proba = GBDT.predict_proba(C_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, GBDT_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, GBDT_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, GBDT_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, GBDT_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, GBDT_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('GBDT_C.svg', dpi=300,format="svg")
plt.show()


# In[234]:


from sklearn.ensemble import GradientBoostingClassifier
GBDT = GradientBoostingClassifier()
GBDT.fit(R_train,y_train)
GBDT_test_pred = GBDT.predict(R_test)
GBDT_test_proba = GBDT.predict_proba(R_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, GBDT_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, GBDT_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, GBDT_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, GBDT_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, GBDT_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('GBDT_R.svg', dpi=300,format="svg")
plt.show()


# In[235]:


from sklearn.ensemble import GradientBoostingClassifier
GBDT = GradientBoostingClassifier()
GBDT.fit(S_train,y_train)
GBDT_test_pred = GBDT.predict(S_test)
GBDT_test_proba = GBDT.predict_proba(S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, GBDT_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, GBDT_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, GBDT_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, GBDT_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, GBDT_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('GBDT_S.svg', dpi=300,format="svg")
plt.show()


# In[236]:


from sklearn.ensemble import GradientBoostingClassifier
GBDT = GradientBoostingClassifier()
GBDT.fit(C_R_train,y_train)
GBDT_test_pred = GBDT.predict(C_R_test)
GBDT_test_proba = GBDT.predict_proba(C_R_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, GBDT_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, GBDT_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, GBDT_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, GBDT_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, GBDT_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('GBDT_C_R.svg', dpi=300,format="svg")
plt.show()


# In[237]:


from sklearn.ensemble import GradientBoostingClassifier
GBDT = GradientBoostingClassifier()
GBDT.fit(C_S_train,y_train)
GBDT_test_pred = GBDT.predict(C_S_test)
GBDT_test_proba = GBDT.predict_proba(C_S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, GBDT_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, GBDT_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, GBDT_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, GBDT_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, GBDT_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('GBDT_C_S.svg', dpi=300,format="svg")
plt.show()


# In[238]:


from sklearn.ensemble import GradientBoostingClassifier
GBDT = GradientBoostingClassifier()
GBDT.fit(R_S_train,y_train)
GBDT_test_pred = GBDT.predict(R_S_test)
GBDT_test_proba = GBDT.predict_proba(R_S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, GBDT_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, GBDT_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, GBDT_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, GBDT_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, GBDT_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('GBDT_R_S.svg', dpi=300,format="svg")
plt.show()


# In[239]:


from sklearn.ensemble import GradientBoostingClassifier
GBDT = GradientBoostingClassifier()
GBDT.fit(ALL_train,y_train)
GBDT_test_pred = GBDT.predict(ALL_test)
GBDT_test_proba = GBDT.predict_proba(ALL_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, GBDT_test_pred)))
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, GBDT_test_pred))) 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, GBDT_test_pred)))
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, GBDT_test_pred)))

fpr,tpr,threshold = roc_curve(y_test, GBDT_test_proba[:,1])
roc_auc = auc(fpr,tpr)
lw=2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 15}
plt.legend(loc="lower right",prop=font1)
plt.savefig('GBDT_ALL.svg', dpi=300,format="svg")
plt.show()


# In[ ]:





# In[ ]:




