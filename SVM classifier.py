#!/usr/bin/env python
# coding: utf-8

# # Train SVM classifier using sklearn digits dataset (i.e. from sklearn.datasets import load_digits) and then
# # 1. Measure the accuracy of your model using different kernels such as RBF,  poly, and linear. 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape


# In[3]:


plt.gray() 
plt.matshow(digits.images[0]) 
plt.show()


# In[4]:


dir(digits)


# In[5]:


digits.feature_names


# In[6]:


digits.target_names


# In[7]:


digits.target


# In[9]:


df=pd.DataFrame(digits.data,digits.target)
df.head()


# In[10]:


df['target']=digits.target
df.head(15)


# In[11]:


df.describe()


# 
# # DATA VISUALIZATION

# In[12]:


from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 12))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))


# In[14]:


plt.figure()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
proj = pca.fit_transform(digits.data)
plt.scatter(proj[:, 0], proj[:, 1], c=digits.target, cmap="Paired")
plt.colorbar()


# In[15]:


sns.set(rc={'figure.figsize':(20,20)})
correlation_matrix = df.corr().round(1)
sns.heatmap(data=correlation_matrix, annot=True,cmap='Pastel1')


# # DATA PREPROCESSING

# In[16]:


df.isna().sum()


# In[21]:


import missingno as msno
msno.bar(df,color="pink")
plt.show()


# # MODELING WITH SVM

# In[23]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[24]:


X=df.drop('target',axis='columns')
y=df.target


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# # RBF KERNEL

# In[26]:


rbf_model = SVC(kernel='rbf',gamma=0.002,probability=True)
rbf_model.fit(X_train,y_train)


# In[28]:


rbf_model.score(X_test,y_test)


# In[29]:


y_pred=rbf_model.predict(X_test)


# In[33]:


import scikitplot as skplt
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.pipeline import Pipeline

print(accuracy_score(y_test, y_pred))


# In[34]:


print(recall_score(y_test, y_pred,average=None))


# In[35]:


print(precision_score(y_test, y_pred,average=None))


# # ROC CURVE

# In[36]:


y_probas = rbf_model.predict_proba(X_test)
skplt.metrics.plot_roc(y_test,y_probas,figsize=(10,6),title_fontsize=14,text_fontsize=12)
plt.show()


# # PRECISION RECALL CURVE

# In[37]:


skplt.metrics.plot_precision_recall(y_test,y_probas,figsize=(8,6),title_fontsize=14,text_fontsize=12)
plt.show()


# # LEARNING CURVE

# In[38]:


skplt.estimators.plot_learning_curve(rbf_model, X,y,figsize=(8,6),title_fontsize=14,text_fontsize=12)
plt.show()


# # CONFUSION MATRIX

# In[39]:


skplt.metrics.plot_confusion_matrix(y_test,y_pred,figsize=(10,6),title_fontsize=14,text_fontsize=12,cmap=plt.cm.PuBu)
plt.show()


# # LINEAR KERNEL

# In[52]:


linear_model = SVC(kernel='linear',C=0.001,probability=True)
linear_model.fit(X_train,y_train)


# In[53]:


linear_model.score(X_test,y_test)


# In[55]:


y_pred=linear_model.predict(X_test)


# In[56]:


import scikitplot as skplt
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.pipeline import Pipeline

print(accuracy_score(y_test, y_pred))


# In[57]:


print(recall_score(y_test, y_pred,average=None))


# In[58]:


print(precision_score(y_test, y_pred,average=None))


# # ROC CURVE

# In[59]:


y_probas = linear_model.predict_proba(X_test)
skplt.metrics.plot_roc(y_test,y_probas,figsize=(10,6),title_fontsize=14,text_fontsize=12)
plt.show()


# # PRECISION RECALL CURVE

# In[60]:


skplt.metrics.plot_precision_recall(y_test,y_probas,figsize=(8,6),title_fontsize=14,text_fontsize=12)
plt.show()


# # LEARNING CURVE

# In[61]:


skplt.estimators.plot_learning_curve(rbf_model, X,y,figsize=(8,6),title_fontsize=14,text_fontsize=12)
plt.show()


# # CONFUSION MATRIX

# In[62]:


skplt.metrics.plot_confusion_matrix(y_test,y_pred,figsize=(10,6),title_fontsize=14,text_fontsize=12,cmap=plt.cm.YlGn)
plt.show()

