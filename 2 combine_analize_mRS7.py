#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import DataFrame
import os


# In[2]:


df = pd.read_csv('./dataset/3.2 stroke_mRS7.csv')
df


# In[3]:


Y = np.array(df.pop('mRS_7'))
X = np.array(df)


# In[4]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler
X = min_max_scaler().fit_transform(X)


# In[5]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3,random_state = 3)


# In[6]:


C_train = x_train[:,-6:]
R_train = x_train[:,:9]
S_train = x_train[:,9]
C_R_train = x_train[:,[10,11,12,13,14,15,0,1,2,3,4,5,6,7,8]]
C_S_train = x_train[:,[10,11,12,13,14,15,9]]
R_S_train = x_train[:,:10]
ALL_train = x_train[:,:]


# In[7]:


print(C_train.shape,
R_train.shape,
S_train.shape,
C_R_train.shape,
C_S_train.shape,
R_S_train.shape,
ALL_train.shape)


# In[8]:


C_test = x_test[:,-6:]
R_test = x_test[:,:9]
S_test = x_test[:,9]
C_R_test = x_test[:,[10,11,12,13,14,15,0,1,2,3,4,5,6,7,8]]
C_S_test = x_test[:,[10,11,12,13,14,15,9]]
R_S_test = x_test[:,:10]
ALL_test = x_test[:,:]


# In[9]:


S_train = S_train.reshape(-1,1)
S_test = S_test.reshape(-1,1)


# <font color=#0099ff  size=5 face="黑体">方法1：DT</font>

# In[47]:


from sklearn import tree 
from sklearn.metrics import accuracy_score,auc,roc_curve
from sklearn import metrics
DT_classifier = tree.DecisionTreeClassifier(criterion="entropy"
                                  ,random_state=27# 
                                  ,splitter="random")
DT_classifier.fit(C_train,y_train)
DT_train_pred = DT_classifier.predict(C_train)
DT_test_pred = DT_classifier.predict(C_test)
DT_train_acc = accuracy_score(y_train, DT_train_pred)  
DT_test_acc = accuracy_score(y_test, DT_test_pred)  
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DT_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DT_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DT_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DT_test_pred,average='macro')))


# In[48]:


from sklearn import tree 
from sklearn.metrics import accuracy_score,auc,roc_curve
from sklearn import metrics
import matplotlib.pyplot as plt
n_classes=7
from sklearn.preprocessing import label_binarize
y_test_label = label_binarize(y_test, classes=[0,1,2,3,4,5,6])
from scipy import interp
from itertools import cycle
DT_test_proba = DT_classifier.predict_proba(C_test)                   #change
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], DT_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), DT_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("DT_ROC_C.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[17]:


DT_classifier = tree.DecisionTreeClassifier(criterion="entropy"
                                  ,random_state=30 # 
                                  ,splitter="random")
DT_classifier.fit(R_train,y_train)
DT_train_pred = DT_classifier.predict(R_train)
DT_test_pred = DT_classifier.predict(R_test)
DT_train_acc = accuracy_score(y_train, DT_train_pred)  
DT_test_acc = accuracy_score(y_test, DT_test_pred)  
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DT_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DT_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DT_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DT_test_pred,average='macro')))


# In[18]:


DT_test_proba = DT_classifier.predict_proba(R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], DT_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), DT_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("DT_ROC_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[19]:


DT_classifier = tree.DecisionTreeClassifier(criterion="entropy"
                                  ,random_state=30 # 
                                  ,splitter="random")
DT_classifier.fit(S_train,y_train)
DT_train_pred = DT_classifier.predict(S_train)
DT_test_pred = DT_classifier.predict(S_test)
DT_train_acc = accuracy_score(y_train, DT_train_pred)  
DT_test_acc = accuracy_score(y_test, DT_test_pred)  
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DT_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DT_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DT_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DT_test_pred,average='macro')))

DT_test_proba = DT_classifier.predict_proba(S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], DT_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), DT_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("DT_ROC_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[20]:


DT_classifier = tree.DecisionTreeClassifier(criterion="entropy"
                                  ,random_state=30 # 
                                  ,splitter="random")
DT_classifier.fit(C_R_train,y_train)
DT_train_pred = DT_classifier.predict(C_R_train)
DT_test_pred = DT_classifier.predict(C_R_test)
DT_train_acc = accuracy_score(y_train, DT_train_pred)  
DT_test_acc = accuracy_score(y_test, DT_test_pred)  
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DT_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DT_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DT_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DT_test_pred,average='macro')))

DT_test_proba = DT_classifier.predict_proba(C_R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], DT_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), DT_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("DT_ROC_C_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[52]:


DT_classifier = tree.DecisionTreeClassifier(criterion="entropy"
                                  ,random_state=30 # 
                                  ,splitter="random")
DT_classifier.fit(C_S_train,y_train)
DT_train_pred = DT_classifier.predict(C_S_train)
DT_test_pred = DT_classifier.predict(C_S_test)
DT_train_acc = accuracy_score(y_train, DT_train_pred)  
DT_test_acc = accuracy_score(y_test, DT_test_pred)  
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DT_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DT_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DT_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DT_test_pred,average='macro')))

DT_test_proba = DT_classifier.predict_proba(C_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], DT_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), DT_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("DT_ROC_C_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[22]:


DT_classifier = tree.DecisionTreeClassifier(criterion="entropy"
                                  ,random_state=30 # 
                                  ,splitter="random")
DT_classifier.fit(R_S_train,y_train)
DT_train_pred = DT_classifier.predict(R_S_train)
DT_test_pred = DT_classifier.predict(R_S_test)
DT_train_acc = accuracy_score(y_train, DT_train_pred)  
DT_test_acc = accuracy_score(y_test, DT_test_pred)  
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DT_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DT_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DT_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DT_test_pred,average='macro')))

DT_test_proba = DT_classifier.predict_proba(R_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], DT_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), DT_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("DT_ROC_R_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[27]:


DT_classifier = tree.DecisionTreeClassifier(criterion="entropy"
                                  ,random_state=3 # 
                                  ,splitter="random")
DT_classifier.fit(ALL_train,y_train)
DT_train_pred = DT_classifier.predict(ALL_train)
DT_test_pred = DT_classifier.predict(ALL_test)
DT_train_acc = accuracy_score(y_train, DT_train_pred)  
DT_test_acc = accuracy_score(y_test, DT_test_pred)  
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DT_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DT_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DT_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DT_test_pred,average='macro')))

DT_test_proba = DT_classifier.predict_proba(ALL_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], DT_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), DT_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("DT_ROC_ALL.svg", dpi=300,format="svg")                                    #change

plt.show()


# <font color=#0099ff  size=5 face="黑体">方法2：SVM 支持向量机</font>

# In[54]:


from sklearn import svm
SVM_classifier = svm.SVC(C=2, kernel='linear',probability=True)
SVM_classifier.fit(C_train, y_train)                                                #change

SVM_train_pred = SVM_classifier.predict(C_train)                                   #change
SVM_test_pred = SVM_classifier.predict(C_test)                                     #change   

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, SVM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, SVM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  SVM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  SVM_test_pred,average='macro')))

SVM_test_proba = SVM_classifier.predict_proba(C_test)                                   #change

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], SVM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), SVM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))


plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("SVM_ROC_C.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[55]:


SVM_classifier = svm.SVC(C=2, kernel='linear',probability=True)
SVM_classifier.fit(R_train, y_train)                                                #change

SVM_train_pred = SVM_classifier.predict(R_train)                                   #change
SVM_test_pred = SVM_classifier.predict(R_test)                                     #change   

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, SVM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, SVM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  SVM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  SVM_test_pred,average='macro')))

SVM_test_proba = SVM_classifier.predict_proba(R_test)                                   #change

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], SVM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), SVM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))


plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("SVM_ROC_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[56]:


SVM_classifier = svm.SVC(C=2, kernel='linear',probability=True)
SVM_classifier.fit(S_train, y_train)                                                #change

SVM_train_pred = SVM_classifier.predict(S_train)                                   #change
SVM_test_pred = SVM_classifier.predict(S_test)                                     #change   

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, SVM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, SVM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  SVM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  SVM_test_pred,average='macro')))

SVM_test_proba = SVM_classifier.predict_proba(S_test)                                   #change

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], SVM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), SVM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))


plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("SVM_ROC_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[57]:


SVM_classifier = svm.SVC(C=2, kernel='linear',probability=True)
SVM_classifier.fit(C_R_train, y_train)                                                #change

SVM_train_pred = SVM_classifier.predict(C_R_train)                                   #change
SVM_test_pred = SVM_classifier.predict(C_R_test)                                     #change   

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, SVM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, SVM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  SVM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  SVM_test_pred,average='macro')))

SVM_test_proba = SVM_classifier.predict_proba(C_R_test)                                   #change

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], SVM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), SVM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))


plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("SVM_ROC_C_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[58]:


SVM_classifier = svm.SVC(C=2, kernel='linear',probability=True)
SVM_classifier.fit(C_S_train, y_train)                                                #change

SVM_train_pred = SVM_classifier.predict(C_S_train)                                   #change
SVM_test_pred = SVM_classifier.predict(C_S_test)                                     #change   

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, SVM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, SVM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  SVM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  SVM_test_pred,average='macro')))

SVM_test_proba = SVM_classifier.predict_proba(C_S_test)                                   #change

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], SVM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), SVM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))


plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("SVM_ROC_C_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[59]:


SVM_classifier = svm.SVC(C=2, kernel='linear',probability=True)
SVM_classifier.fit(R_S_train, y_train)                                                #change

SVM_train_pred = SVM_classifier.predict(R_S_train)                                   #change
SVM_test_pred = SVM_classifier.predict(R_S_test)                                     #change   

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, SVM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, SVM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  SVM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  SVM_test_pred,average='macro')))

SVM_test_proba = SVM_classifier.predict_proba(R_S_test)                                   #change

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], SVM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), SVM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))


plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("SVM_ROC_R_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[20]:


from sklearn import svm
from sklearn import tree 
from sklearn.metrics import accuracy_score,auc,roc_curve
from sklearn import metrics
import matplotlib.pyplot as plt
n_classes=7
from sklearn.preprocessing import label_binarize
y_test_label = label_binarize(y_test, classes=[0,1,2,3,4,5,6])
from scipy import interp
from itertools import cycle
SVM_classifier = svm.SVC(C=2, kernel='linear',probability=True,random_state=3)
SVM_classifier.fit(ALL_train, y_train)                                                #change

SVM_train_pred = SVM_classifier.predict(ALL_train)                                   #change
SVM_test_pred = SVM_classifier.predict(ALL_test)                                     #change   

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, SVM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, SVM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  SVM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  SVM_test_pred,average='macro')))

SVM_test_proba = SVM_classifier.predict_proba(ALL_test)                                   #change

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], SVM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), SVM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))


plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("SVM_ROC_ALL.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[ ]:





# <font color=#0099ff  size=5 face="黑体">方法3：RF随机森林</font>

# In[61]:


from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier(n_estimators = 1, criterion="gini",random_state=3)

RF_classifier.fit(C_train, y_train)                                                        #change
RF_train_pred = RF_classifier.predict(C_train)                                             #change                                        
RF_test_pred = RF_classifier.predict(C_test)                                               #change

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, RF_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, RF_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  RF_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  RF_test_pred,labels=[0,1,2],average='macro')))

RF_test_proba = RF_classifier.predict_proba(C_test)                                        #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): 
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], RF_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), RF_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("RF_C.svg", dpi=300,format="svg")

plt.show()


# In[68]:


RF_classifier = RandomForestClassifier(n_estimators = 5, criterion="gini",random_state=2)

RF_classifier.fit(R_train, y_train)                        #change
RF_train_pred = RF_classifier.predict(R_train)              #change
RF_test_pred = RF_classifier.predict(R_test)                #change

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, RF_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, RF_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  RF_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  RF_test_pred,labels=[0,1,2],average='macro')))

RF_test_proba = RF_classifier.predict_proba(R_test)                                        #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): 
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], RF_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), RF_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("RF_R.svg", dpi=300,format="svg")                      #change

plt.show()


# In[63]:


RF_classifier = RandomForestClassifier(n_estimators = 5, criterion="gini",random_state=2)

RF_classifier.fit(S_train, y_train)                        #change
RF_train_pred = RF_classifier.predict(S_train)              #change
RF_test_pred = RF_classifier.predict(S_test)                #change

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, RF_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, RF_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  RF_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  RF_test_pred,labels=[0,1,2],average='macro')))

RF_test_proba = RF_classifier.predict_proba(S_test)                                        #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): 
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], RF_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), RF_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("RF_S.svg", dpi=300,format="svg")                      #change

plt.show()


# In[64]:


RF_classifier = RandomForestClassifier(n_estimators = 5, criterion="gini",random_state=2)

RF_classifier.fit(C_R_train, y_train)                        #change
RF_train_pred = RF_classifier.predict(C_R_train)              #change
RF_test_pred = RF_classifier.predict(C_R_test)                #change

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, RF_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, RF_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  RF_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  RF_test_pred,labels=[0,1,2],average='macro')))

RF_test_proba = RF_classifier.predict_proba(C_R_test)                                        #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): 
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], RF_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), RF_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("RF_C_R.svg", dpi=300,format="svg")                      #change

plt.show()


# In[65]:


RF_classifier = RandomForestClassifier(n_estimators = 2, criterion="gini",random_state=1)

RF_classifier.fit(C_S_train, y_train)                        #change
RF_train_pred = RF_classifier.predict(C_S_train)              #change
RF_test_pred = RF_classifier.predict(C_S_test)                #change

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, RF_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, RF_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  RF_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  RF_test_pred,labels=[0,1,2],average='macro')))

RF_test_proba = RF_classifier.predict_proba(C_S_test)                                        #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): 
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], RF_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), RF_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("RF_C_S.svg", dpi=300,format="svg")                      #change

plt.show()


# In[66]:


RF_classifier = RandomForestClassifier(n_estimators = 5, criterion="gini",random_state=2)

RF_classifier.fit(R_S_train, y_train)                        #change
RF_train_pred = RF_classifier.predict(R_S_train)              #change
RF_test_pred = RF_classifier.predict(R_S_test)                #change

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, RF_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, RF_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  RF_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  RF_test_pred,labels=[0,1,2],average='macro')))

RF_test_proba = RF_classifier.predict_proba(R_S_test)                                        #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): 
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], RF_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), RF_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("RF_R_S.svg", dpi=300,format="svg")                      #change

plt.show()


# In[67]:


RF_classifier = RandomForestClassifier(n_estimators = 5, criterion="gini",random_state=2)

RF_classifier.fit(ALL_train, y_train)                        #change
RF_train_pred = RF_classifier.predict(ALL_train)              #change
RF_test_pred = RF_classifier.predict(ALL_test)                #change

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, RF_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, RF_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  RF_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  RF_test_pred,labels=[0,1,2],average='macro')))

RF_test_proba = RF_classifier.predict_proba(ALL_test)                                        #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): 
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], RF_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), RF_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("RF_ALL.svg", dpi=300,format="svg")                      #change

plt.show()


# In[ ]:





# <font color=#0099ff  size=5 face="黑体">方法4：BiLSTM</font>

# In[21]:


from sklearn.preprocessing import LabelBinarizer,StandardScaler
Class = [0,1,2,3,4,5,6]
Class_dict = dict(zip(Class, range(len(Class))))
Class_dict
lb = LabelBinarizer()
lb.fit(list(Class_dict.values()))
y_train_labels = lb.transform(y_train)
n_classes=7
y_test_label = label_binarize(y_test, classes=[0,1,2,3,4,5,6])
def Predict(X):
    RNN_test_label = []
    Class = [0,1,2,3,4,5,6]
    Class_dict = dict(zip(Class, range(len(Class))))
    Class_dict
    for i in range(0,X.shape[0]):
        RNN_test_label.append(Class_dict[np.argmax(X[i])])
    RNN_test_label = np.array(RNN_test_label,dtype = 'int64')
    return RNN_test_label


# In[83]:


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
                             input_shape=(1, 6)))                                                           #change
model.add(Bidirectional(LSTM(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_train.reshape(C_train.shape[0],1,C_train.shape[1])                                        #change                            
x_test_lstm = C_test.reshape(C_test.shape[0],1,C_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=100, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))


# In[84]:


# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("LSTM_C.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[74]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 9)))                                                           #change
model.add(Bidirectional(LSTM(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = R_train.reshape(R_train.shape[0],1,R_train.shape[1])                                        #change                            
x_test_lstm = R_test.reshape(R_test.shape[0],1,R_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=500, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("LSTM_R.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[76]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 1)))                                                           #change
model.add(Bidirectional(LSTM(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = S_train.reshape(S_train.shape[0],1,1)                                        #change                            
x_test_lstm = S_test.reshape(S_test.shape[0],1,1)                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=500, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("LSTM_S.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[77]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 15)))                                                           #change
model.add(Bidirectional(LSTM(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_R_train.reshape(C_R_train.shape[0],1,C_R_train.shape[1])                                        #change                            
x_test_lstm = C_R_test.reshape(C_R_test.shape[0],1,C_R_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=500, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("LSTM_C_R.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[78]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 7)))                                                           #change
model.add(Bidirectional(LSTM(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_S_train.reshape(C_S_train.shape[0],1,C_S_train.shape[1])                                        #change                            
x_test_lstm = C_S_test.reshape(C_S_test.shape[0],1,C_S_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=500, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("LSTM_C_S.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[79]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 10)))                                                           #change
model.add(Bidirectional(LSTM(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = R_S_train.reshape(R_S_train.shape[0],1,R_S_train.shape[1])                                        #change                            
x_test_lstm = R_S_test.reshape(R_S_test.shape[0],1,R_S_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=500, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("LSTM_R_S.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[29]:


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
                             input_shape=(1, 16)))                                                           #change
model.add(Bidirectional(LSTM(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = ALL_train.reshape(ALL_train.shape[0],1,ALL_train.shape[1])                                        #change                            
x_test_lstm = ALL_test.reshape(ALL_test.shape[0],1,ALL_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=150, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("LSTM_ALL.svg", dpi=300,format="svg")                                            #change

plt.show()


# <font color=#0099ff  size=5 face="黑体">方法5：BiRNN</font>

# In[85]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 6)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_train.reshape(C_train.shape[0],1,C_train.shape[1])                                        #change                            
x_test_lstm = C_test.reshape(C_test.shape[0],1,C_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=100, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("RNN_C.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[86]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 9)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = R_train.reshape(R_train.shape[0],1,R_train.shape[1])                                        #change                            
x_test_lstm = R_test.reshape(R_test.shape[0],1,R_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=500, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("RNN_R.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[87]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 1)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = S_train.reshape(S_train.shape[0],1,1)                                        #change                            
x_test_lstm = S_test.reshape(S_test.shape[0],1,1)                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=500, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("RNN_S.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[88]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 15)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_R_train.reshape(C_R_train.shape[0],1,C_R_train.shape[1])                                        #change                            
x_test_lstm = C_R_test.reshape(C_R_test.shape[0],1,C_R_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=500, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("RNN_C_R.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[89]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 7)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_S_train.reshape(C_S_train.shape[0],1,C_S_train.shape[1])                                        #change                            
x_test_lstm = C_S_test.reshape(C_S_test.shape[0],1,C_S_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=500, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("RNN_C_S.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[90]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 10)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = R_S_train.reshape(R_S_train.shape[0],1,R_S_train.shape[1])                                        #change                            
x_test_lstm = R_S_test.reshape(R_S_test.shape[0],1,R_S_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=500, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("RNN_R_S.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[34]:


model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(1, 16)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = ALL_train.reshape(ALL_train.shape[0],1,ALL_train.shape[1])                                        #change                            
x_test_lstm = ALL_test.reshape(ALL_test.shape[0],1,ALL_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=150, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("RNN_ALL.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[ ]:





# <font color=#0099ff  size=5 face="黑体">方法6：BiGRU</font>

# In[92]:


model = Sequential()
model.add(Bidirectional(GRU(10, return_sequences=True),
                             input_shape=(1, 6)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_train.reshape(C_train.shape[0],1,C_train.shape[1])                                        #change                            
x_test_lstm = C_test.reshape(C_test.shape[0],1,C_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=100, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("GRU_C.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[93]:


model = Sequential()
model.add(Bidirectional(GRU(10, return_sequences=True),
                             input_shape=(1, 9)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = R_train.reshape(R_train.shape[0],1,R_train.shape[1])                                        #change                            
x_test_lstm = R_test.reshape(R_test.shape[0],1,R_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=500, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("GRU_R.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[94]:


model = Sequential()
model.add(Bidirectional(GRU(10, return_sequences=True),
                             input_shape=(1, 1)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = S_train.reshape(S_train.shape[0],1,1)                                        #change                            
x_test_lstm = S_test.reshape(S_test.shape[0],1,1)                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=500, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("GRU_S.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[95]:


model = Sequential()
model.add(Bidirectional(GRU(10, return_sequences=True),
                             input_shape=(1, 15)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_R_train.reshape(C_R_train.shape[0],1,C_R_train.shape[1])                                        #change                            
x_test_lstm = C_R_test.reshape(C_R_test.shape[0],1,C_R_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=500, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("GRU_C_R.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[96]:


model = Sequential()
model.add(Bidirectional(GRU(10, return_sequences=True),
                             input_shape=(1, 7)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = C_S_train.reshape(C_S_train.shape[0],1,C_S_train.shape[1])                                        #change                            
x_test_lstm = C_S_test.reshape(C_S_test.shape[0],1,C_S_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=500, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("GRU_C_S.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[97]:


model = Sequential()
model.add(Bidirectional(GRU(10, return_sequences=True),
                             input_shape=(1, 10)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = R_S_train.reshape(R_S_train.shape[0],1,R_S_train.shape[1])                                        #change                            
x_test_lstm = R_S_test.reshape(R_S_test.shape[0],1,R_S_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=500, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("GRU_R_S.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[35]:


model = Sequential()
model.add(Bidirectional(GRU(10, return_sequences=True),
                             input_shape=(1, 16)))                                                           #change
model.add(Bidirectional(GRU(10)))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_lstm = ALL_train.reshape(ALL_train.shape[0],1,ALL_train.shape[1])                                        #change                            
x_test_lstm = ALL_test.reshape(ALL_test.shape[0],1,ALL_test.shape[1])                                       #change

history = model.fit(x_train_lstm,y_train_labels, validation_data=(x_test_lstm, y_test_label), epochs=150, verbose=1)

LSTM_train_proba = model.predict(x_train_lstm)
LSTM_train_pred =Predict(LSTM_train_proba)
LSTM_test_proba = model.predict(x_test_lstm)
LSTM_test_pred =Predict(LSTM_test_proba)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, LSTM_test_pred))) 
 
print('macro-PRE:{:.4f}'.format(metrics.precision_score(y_test, LSTM_test_pred,average='macro'))) 
 
print('macro-SEN:{:.4f}'.format(metrics.recall_score(y_test,  LSTM_test_pred,average='macro')))
 
print('macroF1-score:{:.4f}'.format(metrics.f1_score(y_test,  LSTM_test_pred,labels=[0,1,2],average='macro')))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], LSTM_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), LSTM_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("GRU_ALL.svg", dpi=300,format="svg")                                            #change

plt.show()


# In[ ]:





# <font color=#0099ff  size=5 face="黑体">方法7：DA</font>

# In[100]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

DA = LinearDiscriminantAnalysis()
DA.fit(C_train,y_train)
DA_test_pred = DA.predict(C_test)
DA_test_proba = DA.predict_proba(C_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DA_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DA_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DA_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DA_test_pred,average='macro')))

DA_test_proba = DA.predict_proba(C_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], DA_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), DA_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("DA_ROC_C.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[101]:


DA = LinearDiscriminantAnalysis()
DA.fit(R_train,y_train)
DA_test_pred = DA.predict(R_test)
DA_test_proba = DA.predict_proba(R_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DA_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DA_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DA_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DA_test_pred,average='macro')))

DA_test_proba = DA.predict_proba(R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], DA_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), DA_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("DA_ROC_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[102]:


DA = LinearDiscriminantAnalysis()
DA.fit(S_train,y_train)
DA_test_pred = DA.predict(S_test)
DA_test_proba = DA.predict_proba(S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DA_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DA_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DA_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DA_test_pred,average='macro')))

DA_test_proba = DA.predict_proba(S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], DA_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), DA_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("DA_ROC_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[103]:


DA = LinearDiscriminantAnalysis()
DA.fit(C_R_train,y_train)
DA_test_pred = DA.predict(C_R_test)
DA_test_proba = DA.predict_proba(C_R_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DA_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DA_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DA_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DA_test_pred,average='macro')))

DA_test_proba = DA.predict_proba(C_R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], DA_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), DA_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("DA_ROC_C_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[104]:


DA = LinearDiscriminantAnalysis()
DA.fit(C_S_train,y_train)
DA_test_pred = DA.predict(C_S_test)
DA_test_proba = DA.predict_proba(C_S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DA_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DA_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DA_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DA_test_pred,average='macro')))

DA_test_proba = DA.predict_proba(C_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], DA_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), DA_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("DA_ROC_C_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[105]:


DA = LinearDiscriminantAnalysis()
DA.fit(R_S_train,y_train)
DA_test_pred = DA.predict(R_S_test)
DA_test_proba = DA.predict_proba(R_S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DA_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DA_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DA_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DA_test_pred,average='macro')))

DA_test_proba = DA.predict_proba(R_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], DA_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), DA_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("DA_ROC_R_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[106]:


DA = LinearDiscriminantAnalysis()
DA.fit(ALL_train,y_train)
DA_test_pred = DA.predict(ALL_test)
DA_test_proba = DA.predict_proba(ALL_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, DA_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, DA_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, DA_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, DA_test_pred,average='macro')))

DA_test_proba = DA.predict_proba(ALL_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], DA_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), DA_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("DA_ROC_ALL.svg", dpi=300,format="svg")                                    #change

plt.show()


# <font color=#0099ff  size=5 face="黑体">方法8：NB</font>

# In[107]:


from sklearn.naive_bayes import GaussianNB
NB = GaussianNB(var_smoothing=1e-02)
NB.fit(C_train,y_train)
NB_test_pred = NB.predict(C_test)
NB_test_proba = NB.predict_proba(C_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, NB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, NB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, NB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, NB_test_pred,average='macro')))

NB_test_proba = NB.predict_proba(C_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], NB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), NB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("NB_ROC_C.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[108]:


NB = GaussianNB()
NB.fit(R_train,y_train)
NB_test_pred = NB.predict(R_test)
NB_test_proba = NB.predict_proba(R_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, NB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, NB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, NB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, NB_test_pred,average='macro')))

NB_test_proba = NB.predict_proba(R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], NB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), NB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("NB_ROC_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[109]:


NB = GaussianNB()
NB.fit(S_train,y_train)
NB_test_pred = NB.predict(S_test)
NB_test_proba = NB.predict_proba(S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, NB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, NB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, NB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, NB_test_pred,average='macro')))

NB_test_proba = NB.predict_proba(S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], NB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), NB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("NB_ROC_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[110]:


NB = GaussianNB()
NB.fit(C_R_train,y_train)
NB_test_pred = NB.predict(C_R_test)
NB_test_proba = NB.predict_proba(C_R_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, NB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, NB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, NB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, NB_test_pred,average='macro')))

NB_test_proba = NB.predict_proba(C_R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], NB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), NB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("NB_ROC_C_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[111]:


NB = GaussianNB()
NB.fit(C_S_train,y_train)
NB_test_pred = NB.predict(C_S_test)
NB_test_proba = NB.predict_proba(C_S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, NB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, NB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, NB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, NB_test_pred,average='macro')))

NB_test_proba = NB.predict_proba(C_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], NB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), NB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("NB_ROC_C_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[112]:


NB = GaussianNB()
NB.fit(R_S_train,y_train)
NB_test_pred = NB.predict(R_S_test)
NB_test_proba = NB.predict_proba(R_S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, NB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, NB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, NB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, NB_test_pred,average='macro')))

NB_test_proba = NB.predict_proba(R_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], NB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), NB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("NB_ROC_R_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[113]:


NB = GaussianNB(var_smoothing=1e-09)
NB.fit(ALL_train,y_train)
NB_test_pred = NB.predict(ALL_test)
NB_test_proba = NB.predict_proba(ALL_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, NB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, NB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, NB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, NB_test_pred,average='macro')))

NB_test_proba = NB.predict_proba(ALL_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], NB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), NB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("NB_ROC_ALL.svg", dpi=300,format="svg")                                    #change

plt.show()


# <font color=#0099ff  size=5 face="黑体">方法8：KNN</font>

# In[114]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(C_train,y_train)
knn_test_pred = knn.predict(C_test)
knn_test_proba = knn.predict_proba(C_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, knn_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, knn_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, knn_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, knn_test_pred,average='macro')))

knn_test_proba = knn.predict_proba(C_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], knn_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), knn_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("knn_ROC_C.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[115]:


knn = KNeighborsClassifier()
knn.fit(R_train,y_train)
knn_test_pred = knn.predict(R_test)
knn_test_proba = knn.predict_proba(R_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, knn_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, knn_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, knn_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, knn_test_pred,average='macro')))

knn_test_proba = knn.predict_proba(R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], knn_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), knn_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("knn_ROC_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[116]:


knn = KNeighborsClassifier()
knn.fit(S_train,y_train)
knn_test_pred = knn.predict(S_test)
knn_test_proba = knn.predict_proba(S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, knn_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, knn_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, knn_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, knn_test_pred,average='macro')))

knn_test_proba = knn.predict_proba(S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], knn_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), knn_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("knn_ROC_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[117]:


knn = KNeighborsClassifier()
knn.fit(C_R_train,y_train)
knn_test_pred = knn.predict(C_R_test)
knn_test_proba = knn.predict_proba(C_R_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, knn_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, knn_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, knn_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, knn_test_pred,average='macro')))

knn_test_proba = knn.predict_proba(C_R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], knn_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), knn_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("knn_ROC_C_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[118]:


knn = KNeighborsClassifier()
knn.fit(C_S_train,y_train)
knn_test_pred = knn.predict(C_S_test)
knn_test_proba = knn.predict_proba(C_S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, knn_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, knn_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, knn_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, knn_test_pred,average='macro')))

knn_test_proba = knn.predict_proba(C_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], knn_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), knn_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("knn_ROC_C_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[119]:


knn = KNeighborsClassifier()
knn.fit(R_S_train,y_train)
knn_test_pred = knn.predict(R_S_test)
knn_test_proba = knn.predict_proba(R_S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, knn_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, knn_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, knn_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, knn_test_pred,average='macro')))

knn_test_proba = knn.predict_proba(R_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], knn_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), knn_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("knn_ROC_R_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[120]:


knn = KNeighborsClassifier()
knn.fit(ALL_train,y_train)
knn_test_pred = knn.predict(ALL_test)
knn_test_proba = knn.predict_proba(ALL_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, knn_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, knn_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, knn_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, knn_test_pred,average='macro')))

knn_test_proba = knn.predict_proba(ALL_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], knn_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), knn_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("knn_ROC_ALL.svg", dpi=300,format="svg")                                    #change

plt.show()


# <font color=#0099ff  size=5 face="黑体">方法9：MLP</font>

# In[121]:


from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(hidden_layer_sizes=(100,20),random_state=1)
MLP.fit(C_train,y_train)
MLP_test_pred = MLP.predict(C_test)
MLP_test_proba = MLP.predict_proba(C_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, MLP_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, MLP_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, MLP_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, MLP_test_pred,average='macro')))

MLP_test_proba = MLP.predict_proba(C_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], MLP_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), MLP_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("MLP_ROC_C.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[122]:


from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(hidden_layer_sizes=(400,200),random_state=1)
MLP.fit(R_train,y_train)
MLP_test_pred = MLP.predict(R_test)
MLP_test_proba = MLP.predict_proba(R_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, MLP_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, MLP_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, MLP_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, MLP_test_pred,average='macro')))

MLP_test_proba = MLP.predict_proba(R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], MLP_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), MLP_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("MLP_ROC_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[123]:


MLP = MLPClassifier(hidden_layer_sizes=(400,200),random_state=1)
MLP.fit(S_train,y_train)
MLP_test_pred = MLP.predict(S_test)
MLP_test_proba = MLP.predict_proba(S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, MLP_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, MLP_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, MLP_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, MLP_test_pred,average='macro')))

MLP_test_proba = MLP.predict_proba(S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], MLP_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), MLP_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("MLP_ROC_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[124]:


MLP = MLPClassifier(hidden_layer_sizes=(400,200),random_state=1)
MLP.fit(C_R_train,y_train)
MLP_test_pred = MLP.predict(C_R_test)
MLP_test_proba = MLP.predict_proba(C_R_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, MLP_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, MLP_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, MLP_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, MLP_test_pred,average='macro')))

MLP_test_proba = MLP.predict_proba(C_R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], MLP_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), MLP_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("MLP_ROC_C_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[125]:


MLP = MLPClassifier(hidden_layer_sizes=(400,200),random_state=1)
MLP.fit(C_S_train,y_train)
MLP_test_pred = MLP.predict(C_S_test)
MLP_test_proba = MLP.predict_proba(C_S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, MLP_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, MLP_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, MLP_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, MLP_test_pred,average='macro')))

MLP_test_proba = MLP.predict_proba(C_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], MLP_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), MLP_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("MLP_ROC_C_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[126]:


MLP = MLPClassifier(hidden_layer_sizes=(400,300),random_state=1)
MLP.fit(R_S_train,y_train)
MLP_test_pred = MLP.predict(R_S_test)
MLP_test_proba = MLP.predict_proba(R_S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, MLP_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, MLP_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, MLP_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, MLP_test_pred,average='macro')))

MLP_test_proba = MLP.predict_proba(R_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], MLP_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), MLP_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("MLP_ROC_R_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[127]:


MLP = MLPClassifier(hidden_layer_sizes=(500,250),random_state=1)
MLP.fit(ALL_train,y_train)
MLP_test_pred = MLP.predict(ALL_test)
MLP_test_proba = MLP.predict_proba(ALL_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, MLP_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, MLP_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, MLP_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, MLP_test_pred,average='macro')))

MLP_test_proba = MLP.predict_proba(ALL_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], MLP_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), MLP_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("MLP_ROC_ALL.svg", dpi=300,format="svg")                                    #change

plt.show()


# <font color=#0099ff  size=10 face="黑体">方法10：BAGGING</font>

# In[128]:


from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier()
bag.fit(C_train,y_train)
bag_test_pred = bag.predict(C_test)
bag_test_proba = bag.predict_proba(C_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, bag_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, bag_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, bag_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, bag_test_pred,average='macro')))

bag_test_proba = bag.predict_proba(C_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], bag_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), bag_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("bag_ROC_C.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[129]:


bag = BaggingClassifier()
bag.fit(R_train,y_train)
bag_test_pred = bag.predict(R_test)
bag_test_proba = bag.predict_proba(R_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, bag_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, bag_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, bag_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, bag_test_pred,average='macro')))

bag_test_proba = bag.predict_proba(R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], bag_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), bag_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("bag_ROC_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[130]:


bag = BaggingClassifier()
bag.fit(S_train,y_train)
bag_test_pred = bag.predict(S_test)
bag_test_proba = bag.predict_proba(S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, bag_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, bag_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, bag_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, bag_test_pred,average='macro')))

bag_test_proba = bag.predict_proba(S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], bag_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), bag_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("bag_ROC_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[131]:


bag = BaggingClassifier()
bag.fit(C_R_train,y_train)
bag_test_pred = bag.predict(C_R_test)
bag_test_proba = bag.predict_proba(C_R_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, bag_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, bag_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, bag_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, bag_test_pred,average='macro')))

bag_test_proba = bag.predict_proba(C_R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], bag_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), bag_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("bag_ROC_C_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[132]:


bag = BaggingClassifier()
bag.fit(C_S_train,y_train)
bag_test_pred = bag.predict(C_S_test)
bag_test_proba = bag.predict_proba(C_S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, bag_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, bag_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, bag_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, bag_test_pred,average='macro')))

bag_test_proba = bag.predict_proba(C_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], bag_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), bag_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("bag_ROC_C_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[133]:


bag = BaggingClassifier()
bag.fit(R_S_train,y_train)
bag_test_pred = bag.predict(R_S_test)
bag_test_proba = bag.predict_proba(R_S_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, bag_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, bag_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, bag_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, bag_test_pred,average='macro')))

bag_test_proba = bag.predict_proba(R_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], bag_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), bag_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("bag_ROC_R_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[48]:


from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(random_state= 13)
bag.fit(ALL_train,y_train)
bag_test_pred = bag.predict(ALL_test)
bag_test_proba = bag.predict_proba(ALL_test)
print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, bag_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, bag_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, bag_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, bag_test_pred,average='macro')))

bag_test_proba = bag.predict_proba(ALL_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], bag_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), bag_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("bag_ROC_ALL.svg", dpi=300,format="svg")                                    #change

plt.show()


# <font color=#0099ff  size=10 face="黑体">方法11：ADBoostING</font>

# In[135]:


from sklearn.ensemble import AdaBoostClassifier
ADB = AdaBoostClassifier()
ADB.fit(C_train,y_train)
ADB_test_pred = ADB.predict(C_test)
ADB_test_proba = ADB.predict_proba(C_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, ADB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, ADB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, ADB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, ADB_test_pred,average='macro')))

ADB_test_proba = ADB.predict_proba(C_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], ADB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), ADB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("ADB_ROC_C.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[136]:


ADB = AdaBoostClassifier()
ADB.fit(R_train,y_train)
ADB_test_pred = ADB.predict(R_test)
ADB_test_proba = ADB.predict_proba(R_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, ADB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, ADB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, ADB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, ADB_test_pred,average='macro')))

ADB_test_proba = ADB.predict_proba(R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], ADB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), ADB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("ADB_ROC_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[137]:


ADB = AdaBoostClassifier()
ADB.fit(S_train,y_train)
ADB_test_pred = ADB.predict(S_test)
ADB_test_proba = ADB.predict_proba(S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, ADB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, ADB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, ADB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, ADB_test_pred,average='macro')))

ADB_test_proba = ADB.predict_proba(S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], ADB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), ADB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("ADB_ROC_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[138]:


ADB = AdaBoostClassifier()
ADB.fit(C_R_train,y_train)
ADB_test_pred = ADB.predict(C_R_test)
ADB_test_proba = ADB.predict_proba(C_R_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, ADB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, ADB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, ADB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, ADB_test_pred,average='macro')))

ADB_test_proba = ADB.predict_proba(C_R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], ADB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), ADB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("ADB_ROC_C_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[139]:


ADB = AdaBoostClassifier()
ADB.fit(C_S_train,y_train)
ADB_test_pred = ADB.predict(C_S_test)
ADB_test_proba = ADB.predict_proba(C_S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, ADB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, ADB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, ADB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, ADB_test_pred,average='macro')))

ADB_test_proba = ADB.predict_proba(C_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], ADB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), ADB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("ADB_ROC_C_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[140]:


ADB = AdaBoostClassifier()
ADB.fit(R_S_train,y_train)
ADB_test_pred = ADB.predict(R_S_test)
ADB_test_proba = ADB.predict_proba(R_S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, ADB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, ADB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, ADB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, ADB_test_pred,average='macro')))

ADB_test_proba = ADB.predict_proba(R_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], ADB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), ADB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("ADB_ROC_R_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[60]:


from sklearn.ensemble import AdaBoostClassifier
ADB = AdaBoostClassifier(random_state= 100)
ADB.fit(ALL_train,y_train)
ADB_test_pred = ADB.predict(ALL_test)
ADB_test_proba = ADB.predict_proba(ALL_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, ADB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, ADB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, ADB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, ADB_test_pred,average='macro')))

ADB_test_proba = ADB.predict_proba(ALL_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], ADB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), ADB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("ADB_ROC_ALL.svg", dpi=300,format="svg")                                    #change

plt.show()


# <font color=#0099ff  size=10 face="黑体">方法11：XGBoostING</font>

# In[142]:


from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(C_train,y_train)
XGB_test_pred = XGB.predict(C_test)
XGB_test_proba = XGB.predict_proba(C_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, XGB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, XGB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, XGB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, XGB_test_pred,average='macro')))

XGB_test_proba = XGB.predict_proba(C_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], XGB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), XGB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("XGB_ROC_C.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[143]:


from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(R_train,y_train)
XGB_test_pred = XGB.predict(R_test)
XGB_test_proba = XGB.predict_proba(R_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, XGB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, XGB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, XGB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, XGB_test_pred,average='macro')))

XGB_test_proba = XGB.predict_proba(R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], XGB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), XGB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("XGB_ROC_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[144]:


from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(S_train,y_train)
XGB_test_pred = XGB.predict(S_test)
XGB_test_proba = XGB.predict_proba(S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, XGB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, XGB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, XGB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, XGB_test_pred,average='macro')))

XGB_test_proba = XGB.predict_proba(S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], XGB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), XGB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("XGB_ROC_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[145]:


from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(C_R_train,y_train)
XGB_test_pred = XGB.predict(C_R_test)
XGB_test_proba = XGB.predict_proba(C_R_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, XGB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, XGB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, XGB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, XGB_test_pred,average='macro')))

XGB_test_proba = XGB.predict_proba(C_R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], XGB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), XGB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("XGB_ROC_C_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[146]:


from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(C_S_train,y_train)
XGB_test_pred = XGB.predict(C_S_test)
XGB_test_proba = XGB.predict_proba(C_S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, XGB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, XGB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, XGB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, XGB_test_pred,average='macro')))

XGB_test_proba = XGB.predict_proba(C_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], XGB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), XGB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("XGB_ROC_C_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[147]:


from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(R_S_train,y_train)
XGB_test_pred = XGB.predict(R_S_test)
XGB_test_proba = XGB.predict_proba(R_S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, XGB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, XGB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, XGB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, XGB_test_pred,average='macro')))

XGB_test_proba = XGB.predict_proba(R_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], XGB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), XGB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("XGB_ROC_R_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[148]:


from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(ALL_train,y_train)
XGB_test_pred = XGB.predict(ALL_test)
XGB_test_proba = XGB.predict_proba(ALL_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, XGB_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, XGB_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, XGB_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, XGB_test_pred,average='macro')))

XGB_test_proba = XGB.predict_proba(ALL_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], XGB_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), XGB_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("XGB_ROC_ALL.svg", dpi=300,format="svg")                                    #change

plt.show()


# <font color=#0099ff  size=10 face="黑体">方法12：GBDT</font>

# In[155]:


from sklearn.ensemble import GradientBoostingClassifier
GBDT = GradientBoostingClassifier()
GBDT.fit(C_train,y_train)
GBDT_test_pred = GBDT.predict(C_test)
GBDT_test_proba = GBDT.predict_proba(C_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, GBDT_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, GBDT_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, GBDT_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, GBDT_test_pred,average='macro')))

GBDT_test_proba = GBDT.predict_proba(C_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], GBDT_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), GBDT_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("GBDT_ROC_C.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[150]:


from sklearn.ensemble import GradientBoostingClassifier
GBDT = GradientBoostingClassifier()
GBDT.fit(R_train,y_train)
GBDT_test_pred = GBDT.predict(R_test)
GBDT_test_proba = GBDT.predict_proba(R_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, GBDT_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, GBDT_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, GBDT_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, GBDT_test_pred,average='macro')))

GBDT_test_proba = GBDT.predict_proba(R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], GBDT_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), GBDT_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("GBDT_ROC_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[151]:


from sklearn.ensemble import GradientBoostingClassifier
GBDT = GradientBoostingClassifier()
GBDT.fit(S_train,y_train)
GBDT_test_pred = GBDT.predict(S_test)
GBDT_test_proba = GBDT.predict_proba(S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, GBDT_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, GBDT_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, GBDT_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, GBDT_test_pred,average='macro')))

GBDT_test_proba = GBDT.predict_proba(S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], GBDT_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), GBDT_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("GBDT_ROC_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[152]:


from sklearn.ensemble import GradientBoostingClassifier
GBDT = GradientBoostingClassifier()
GBDT.fit(C_R_train,y_train)
GBDT_test_pred = GBDT.predict(C_R_test)
GBDT_test_proba = GBDT.predict_proba(C_R_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, GBDT_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, GBDT_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, GBDT_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, GBDT_test_pred,average='macro')))

GBDT_test_proba = GBDT.predict_proba(C_R_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], GBDT_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), GBDT_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("GBDT_ROC_C_R.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[153]:


from sklearn.ensemble import GradientBoostingClassifier
GBDT = GradientBoostingClassifier()
GBDT.fit(C_S_train,y_train)
GBDT_test_pred = GBDT.predict(C_S_test)
GBDT_test_proba = GBDT.predict_proba(C_S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, GBDT_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, GBDT_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, GBDT_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, GBDT_test_pred,average='macro')))

GBDT_test_proba = GBDT.predict_proba(C_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], GBDT_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), GBDT_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("GBDT_ROC_C_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[156]:


from sklearn.ensemble import GradientBoostingClassifier
GBDT = GradientBoostingClassifier()
GBDT.fit(R_S_train,y_train)
GBDT_test_pred = GBDT.predict(R_S_test)
GBDT_test_proba = GBDT.predict_proba(R_S_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, GBDT_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, GBDT_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, GBDT_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, GBDT_test_pred,average='macro')))

GBDT_test_proba = GBDT.predict_proba(R_S_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], GBDT_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), GBDT_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("GBDT_ROC_R_S.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[64]:


from sklearn.ensemble import GradientBoostingClassifier
GBDT = GradientBoostingClassifier(random_state=1)
GBDT.fit(ALL_train,y_train)
GBDT_test_pred = GBDT.predict(ALL_test)
GBDT_test_proba = GBDT.predict_proba(ALL_test)

print('ACC:{:.4f}'.format( metrics.accuracy_score(y_test, GBDT_test_pred))) 
 
print('macro-PRE: {:.4f}'.format(metrics.precision_score(y_test, GBDT_test_pred,average='macro'))) 
 
print('macro-SEN: {:.4f}'.format(metrics.recall_score(y_test, GBDT_test_pred,average='macro')))
 
print('macroF1-score: {:.4f}'.format(metrics.f1_score(y_test, GBDT_test_pred,average='macro')))

GBDT_test_proba = GBDT.predict_proba(ALL_test)                   #change

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], GBDT_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), GBDT_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
# plt.figure()
plt.figure(plt.figure(figsize=(8, 6)))

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.yticks(fontproperties='Times New Roman', size=16,weight='bold')#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=16,weight='bold')
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')
plt.ylabel('True Positive Rate',fontproperties='Times New Roman', size=20,weight='bold')

font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 11}

plt.legend(loc="lower right",prop=font1)

plt.savefig("GBDT_ROC_ALL.svg", dpi=300,format="svg")                                    #change

plt.show()


# In[ ]:





# In[ ]:




