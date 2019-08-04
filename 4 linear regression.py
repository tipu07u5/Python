%matplotlib inline

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import load_boston
boston = load_boston()

boston.keys()

boston.data.shape

print(boston.feature_names)

print(boston.DESCR)

bos = pd.DataFrame(boston.data)
bos.head()

bos.columns = boston.feature_names
bos.head()

boston.target[:5]

bos['PRICE'] = boston.target
bos.head()

#sci-kit learn library

from sklearn.linear_model import LinearRegression
X = bos.drop('PRICE', axis = 1)
lm = LinearRegression()

lm.fit(X,bos.PRICE)

print('Estimated intercept coefficient :', lm.intercept_)
print('Number of coefficients :' , len(lm.coef_))

pd.DataFrame(list(zip(X.columns, lm.coef_)), columns = ['features' , 'estimatedCoefficients'])

plt.scatter(bos.RM, bos.PRICE)
plt.xlabel("Average number of rooms per dwelling (RM)")
plt.ylabel("Housing Price")
plt.title("Relationship between RM and Price")
plt.show()

lm.predict(X)[0:5]

plt.scatter(bos.PRICE, lm.predict(X))
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices : $\hat{Y}_i$")
plt.title("Prices vs Predicted Prices: $Y_i$ vs $\hat{Y}_i$")

mseFull = np.mean((bos.PRICE - lm.predict(X)) ** 2) 
print (mseFull)

lm= LinearRegression()
lm.fit(X[['PTRATIO']], bos.PRICE)

msePTRATIO = np.mean((bos.PRICE - lm.predict(X[['PTRATIO']])) ** 2) 
print (msePTRATIO)

X_train = X[:-50]
X_test = X[-50:]
Y_train = bos.PRICE[:-50]
Y_test = bos.PRICE[-50:]
print (X_train.shape)
print (X_test.shape)
print (Y_train.shape)
print (Y_test.shape)

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, bos.PRICE, test_size=0.33,random_state = 5)
print (X_train.shape)
print (X_test.shape)
print (Y_train.shape)
print (Y_test.shape)

lm = LinearRegression()
lm.fit(X_train, Y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)

print("Fit a model X_train,and calculate MSE with Y_train:",np.mean((Y_train - lm.predict(X_train)) ** 2))
print("Fit a model X_train,and calculate MSE with X_test,Y_test:",np.mean((Y_test - lm.predict(X_test)) ** 2))

plt.scatter(lm.predict(X_train), lm.predict(X_train) - Y_train, c='b' , s=40, alpha=0.5)
plt.scatter(lm.predict(X_test), lm.predict(X_test) - Y_test, c='g' , s=40)
plt.hlines(y = 0, xmin=0, xmax = 50)
plt.title('Residual Plot using training (blue) and test (green) data')
plt.ylabel('Residuals')
