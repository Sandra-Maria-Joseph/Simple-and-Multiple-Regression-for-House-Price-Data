#!/usr/bin/env python
# coding: utf-8

# ### IMPORTING LIBRARIES

# In[1]:


import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score


# ### IMPORTING VISUALIZATION LIBRARIES

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# ### Reading Dataset

# In[3]:


df = pd.read_csv(r'E:\MDS_SEM2\New folder\Machine_Learning\Lab\program_3\kc_house_data.csv')

df.head()


# ### Data Cleaning

# In[4]:


df.drop(['id','date'],axis=1,inplace=True)
df


# ### Exploratory Data Analysis

# In[5]:


df.shape


# There are 21613 rows and 19 columns

# In[6]:


df.info()


# 
# * **Price, Bathrooms,floors,lat and long are of float datatype.**
# * **Other 14 Attributes are of Integer type.**

# In[7]:


df.describe()


# In[8]:


df1=df[['price', 'bedrooms', 'bathrooms', 'sqft_living',
    'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
    'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
    'lat', 'long', 'sqft_living15', 'sqft_lot15']]
h = df1.hist(bins=25,figsize=(16,16),xlabelsize='10',ylabelsize='10',xrot=-15)
sns.despine(left=True, bottom=True)
[x.title.set_size(12) for x in h.ravel()];
[x.yaxis.tick_left() for x in h.ravel()];


# ## Correlation Matrix

# In[9]:


features = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',
            'view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated',
            'zipcode','lat','long','sqft_living15','sqft_lot15']

mask = np.zeros_like(df[features].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(df[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", #"BuGn_r" to reverse 
            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});


# ### <FONT COLOR='BLUE'>**Sqft_living vs Price**</FONT>

# In[10]:


sns.jointplot(x='sqft_living',y='price',data=df1,color='black')


# ### <FONT COLOR='BLUE'>**Grade vs Price**</FONT>

# In[11]:


sns.jointplot(x='grade',y='price',data=df1,color='blue')


# ### <FONT COLOR='BLUE'>**Sqft_above vs Price**</FONT>

# In[12]:


sns.jointplot(x='sqft_above',y='price',data=df1,color='green')


# In[13]:


sns.pairplot(df1)


# ## Modelling

# ### <font color='red'>SIMPLE LINEAR REGRESSION</FONT>

# **Regression is one of the most important methods because it gives us more insight about the data.** When we ask why, it is easier to interpret the relation between the response and explanatory variables.
# 
# Simple Linear Regression is of the form:
# **y=b0+b1x**

# In[14]:


##train test splitting

train_data,test_data = train_test_split(df,train_size = 0.8,random_state=3)
lr = linear_model.LinearRegression()


# In[15]:


X_train = np.array(train_data['sqft_living'], dtype=pd.Series).reshape(-1,1)
y_train = np.array(train_data['price'], dtype=pd.Series)
lr.fit(X_train,y_train)


# In[16]:


X_test = np.array(test_data['sqft_living'], dtype=pd.Series).reshape(-1,1)
y_test = np.array(test_data['price'], dtype=pd.Series)


# In[17]:


pred = lr.predict(X_test)


# In[19]:


print ("Average Price for Test Data: {:.3f}".format(y_test.mean()))
print('Intercept: {}'.format(lr.intercept_))
print('Coefficient: {}'.format(lr.coef_))


# **PRICE= -47235.811 + 282.246*SQFT_LIVING**

# In[27]:


#TRAINING ACCURACY

lr.score(X_train,y_train)


# In[28]:


#TESTING ACCURACY

lr.score(X_test,y_test)


# ##### An empty dataframe is defined. 
# 
# 
# *This dataframe includes Root Mean Squared Error (RMSE), R-squared, Adjusted R-squared and mean of the R-squared values obtained by the k-Fold Cross Validation, which are the important metrics to compare different models. Having a R-squared value closer to one and smaller RMSE means a better fit.*

# In[20]:


evaluation = pd.DataFrame({'Model': [],
                           'Details':[],
                           'Root Mean Squared Error (RMSE)':[],
                           'R-squared (training)':[],
                           'Adjusted R-squared (training)':[],
                           'R-squared (test)':[],
                           'Adjusted R-squared (test)':[],
                           '5-Fold Cross Validation':[]})


# In[18]:


rmsesm = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))
rtrsm = float(format(lr.score(X_train, y_train),'.3f'))
rtesm = float(format(lr.score(X_test, y_test),'.3f'))
cv = float(format(cross_val_score(lr,df[['sqft_living']],df['price'],cv=5).mean(),'.3f'))


# In[21]:


r = evaluation.shape[0]
evaluation.loc[r] = ['Simple Linear Regression','-',rmsesm,rtrsm,'-',rtesm,'-',cv]
evaluation


# In[22]:


plt.figure(figsize=(6.5,5))
plt.scatter(X_test,y_test,color='darkgreen',label="Data", alpha=.1)
plt.plot(X_test,lr.predict(X_test),color="red",label="Predicted Regression Line")
plt.xlabel("Living Space (sqft)", fontsize=15)
plt.ylabel("Price ($)", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend()

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)


# # <span id="3"></span> Defining a Function to Calculate the Adjusted $R^{2}$

# The R-squared increases when the number of features increase. Because of this, sometimes a more robust evaluator is preferred to compare the performance between different models. This evaluater is called adjusted R-squared and it only increases, if the addition of the variable reduces the MSE. The definition of the adjusted $R^{2}$ is:
# 
# $\bar{R^{2}}=R^{2}-\frac{k-1}{n-k}(1-R^{2})$
# 
# where $n$ is the number of observations and $k$ is the number of parameters. 

# In[23]:


def adjustedR2(r2,n,k):
    return r2-(k-1)/(n-k)*(1-r2)


# ## <font color='red'> MULTIPLE REGRESSION 1</FONT>

# I determined ***features*** at first sight by looking at the previous sections and used in my first multiple linear regression. As in the simple regression, I printed the coefficients which the model uses for the predictions. However, this time we must use the below definition for our predictions, if we want to make calculations manually.
# 
# $$h_{\theta}(X)=\theta_{0}+\theta_{1}x_{1}+\theta_{2}x_{2}+...+\theta_{n}x_{n}$$

# In[24]:


train_data_dm,test_data_dm = train_test_split(df,train_size = 0.8,random_state=3)

features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']
complex_model_1 = linear_model.LinearRegression()
complex_model_1.fit(train_data_dm[features],train_data_dm['price'])

print('Intercept: {}'.format(complex_model_1.intercept_))
print('Coefficients: {}'.format(complex_model_1.coef_))

pred = complex_model_1.predict(test_data_dm[features])
rmsecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['price'],pred)),'.3f'))
rtrcm = float(format(complex_model_1.score(train_data_dm[features],train_data_dm['price']),'.3f'))
artrcm = float(format(adjustedR2(complex_model_1.score(train_data_dm[features],train_data_dm['price']),train_data_dm.shape[0],len(features)),'.3f'))
rtecm = float(format(complex_model_1.score(test_data_dm[features],test_data_dm['price']),'.3f'))
artecm = float(format(adjustedR2(complex_model_1.score(test_data_dm[features],test_data['price']),test_data_dm.shape[0],len(features)),'.3f'))
cv = float(format(cross_val_score(complex_model_1,df[features],df['price'],cv=5).mean(),'.3f'))

r = evaluation.shape[0]
evaluation.loc[r] = ['Multiple Regression-1','selected features',rmsecm,rtrcm,artrcm,rtecm,artecm,cv]
evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)


# ## <font color='red'> MULTIPLE REGRESSION 2</FONT>

# In[25]:


features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view',
            'condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated',
            'zipcode','lat','long','sqft_living15','sqft_lot15']
complex_model_2 = linear_model.LinearRegression()
complex_model_2.fit(train_data_dm[features],train_data_dm['price'])

print('Intercept: {}'.format(complex_model_2.intercept_))
print('Coefficients: {}'.format(complex_model_2.coef_))

pred = complex_model_2.predict(test_data_dm[features])

rmsecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['price'],pred)),'.3f'))
rtrcm = float(format(complex_model_2.score(train_data_dm[features],train_data_dm['price']),'.3f'))
artrcm = float(format(adjustedR2(complex_model_2.score(train_data_dm[features],train_data_dm['price']),train_data_dm.shape[0],len(features)),'.3f'))
rtecm = float(format(complex_model_2.score(test_data_dm[features],test_data_dm['price']),'.3f'))
artecm = float(format(adjustedR2(complex_model_2.score(test_data_dm[features],test_data_dm['price']),test_data_dm.shape[0],len(features)),'.3f'))
cv = float(format(cross_val_score(complex_model_2,df[features],df['price'],cv=5).mean(),'.3f'))

r = evaluation.shape[0]
evaluation.loc[r] = ['Multiple Regression-2','selected features',rmsecm,rtrcm,artrcm,rtecm,artecm,cv]
evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)


# ### CONCLUSION

# 🔓 *The Accuracy is less in the case of Simple Linear Regression Model.*
# 
# 🔓 *As the variable increases,the Root Mean Square Error decreases.*
# 
# 🔓 *As the variable increases,the adjusted R squared increases.*
# 
