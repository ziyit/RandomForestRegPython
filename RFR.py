
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
df = pd.read_csv('/Users/ziyit/Desktop/Project2/data.csv',header=0,sep=',')

X = df.iloc[:, :-1].values
y = df['SALES_2016'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=1)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1000, criterion='mse',random_state=1,n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred=forest.predict(X_test)


print('MSE train: %.3f, test: %.3f'%(
mean_squared_error(y_train, y_train_pred),
mean_squared_error(y_test,y_test_pred)))


print('R^2: %.3f, test: %.3f'%(
r2_score(y_train, y_train_pred),
r2_score(y_test,y_test_pred)))



plt.scatter(y_train_pred,y_train_pred-y_train,c='black',marker='o',s=35,alpha=0.5,label='Training data')

plt.scatter(y_test_pred,y_test_pred-y_test,c='lightgreen',marker='o',s=35,alpha=0.5,label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=20000000,lw=2,color='red')
plt.xlim([-10,20000000])
plt.show()

importances = forest.feature_importances_
forest.get_params

feature_names = df.columns
importances = forest.feature_importances_ 
important_names = feature_names[importances > np.mean(importances)] 
print (important_names)



forest.predict(X)








