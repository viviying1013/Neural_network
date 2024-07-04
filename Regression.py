# import library
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# load the diabetes dataset: 
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

## check the shape of the data. 查看原始資料集的形狀
print("Feature data shape:", diabetes_X.shape)  # 特徵數據形狀:(442, 10)
print("Target data shape:", diabetes_y.shape)  # 目標變量形狀:(442,)


# Use only one feature(the 3rd one):
"""
use numpy.newaxis to add a new axis to the array. 
    to make sure it is a 2D array.
the reason to use only one feature:
    1. simplfied the question.簡化問題
    2. descending dimension.降維
    3. understanding the effect of each feature on the prediction.了解每個特徵對模型預測的影響 
"""
diabetes_X = diabetes_X[:, np.newaxis, 2]

# split dataset into training set and test set
diabetes_X_train = diabetes_X[:-20] #train set: choose all but the last 20 samples 
diabetes_X_test = diabetes_X[-20:] #test set: choose the last 20 samples
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# create linear regression model
"""
建立一個線性回歸模型`linear_model.LinearRegression()` ，並使用 diabetes_X_train 和 diabetes_y_train 的訓練資料來訓練模型。
"""
## Create linear regression object
regr = linear_model.LinearRegression() 
## Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train) 
## Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)


# make predictions by using training set
## 顯示模型的係數(斜率),均方誤差(MSE)和決定係數(R²). print the coefficients(slopes), mean squared error(MSE) and coefficient of determination(R²).
print('Coefficients: \n', regr.coef_) # \n:換行
print('Mean squared error: %.2f' % mean_squared_error(diabetes_y_test, diabetes_y_pred)) # %.2f: 保留兩位小數
print('Coefficient of determination: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black') # 散點圖
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=1.5) # 線圖
plt.xticks(()) 
plt.yticks(())
plt.show()
