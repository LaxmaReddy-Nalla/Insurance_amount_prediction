import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('insurance.csv')
df.head()
df.info()

X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_X = LabelEncoder()
X[:,1] = le_X.fit_transform(X[:,1])
X[:,4] = le_X.fit_transform(X[:,4])
X[:,5] = le_X.fit_transform(X[:,5])

ohe = OneHotEncoder(categorical_features=[5])
X = ohe.fit_transform(X).toarray() 
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)

pred_y = model.predict(X_test)
model.score(X_train,Y_train)
model.score(X_test,Y_test)

X = np.append(arr=np.ones((1338,1)).astype(int),values = X,axis =1)

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5,6,7,8]]
reg_ols = sm.OLS(endog=Y,exog=X_opt).fit()
reg_ols.summary()

X_opt = X[:,[0,1,2,3,4,5,7,8]]
reg_ols = sm.OLS(endog=Y,exog=X_opt).fit()
reg_ols.summary()

X_opt = X[:,[0,1,2,3,4,7,8]]
reg_ols = sm.OLS(endog=Y,exog=X_opt).fit()
reg_ols.summary()

X_opt = X[:,[0,2,3,4,7,8]]
reg_ols = sm.OLS(endog=Y,exog=X_opt).fit()
reg_ols.summary()

X_opt = X[:,[0,2,4,7,8]]
reg_ols = sm.OLS(endog=Y,exog=X_opt).fit()
reg_ols.summary()

X_opt = X[:,[0,4,7,8]]
reg_ols = sm.OLS(endog=Y,exog=X_opt).fit()
reg_ols.summary()

X_opt = X[:,[0,4,8]]
reg_ols = sm.OLS(endog=Y,exog=X_opt).fit()
reg_ols.summary()