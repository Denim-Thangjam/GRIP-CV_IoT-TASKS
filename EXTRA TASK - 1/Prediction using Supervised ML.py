import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline

df = pd.read_csv("http://bit.ly/w-data")
df.head()
df.shape

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

plt.scatter(X, y)
plt.title('Raw Data Plot')
plt.xlabel('Time Studied (in Hrs)')
plt.ylabel('Marks (in %)')
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train, y_train)

print(f'Coefficient : {LR.coef_}\nIntercept : {LR.intercept_}\nAccuracy : {round(LR.score(X_test, y_test),4)*100}%')

pred = LR.predict(X_test)
df_pred = pd.DataFrame({'Actual': y_test, 'Predicted' : pred})
df_pred

plt.plot(X_test, pred, 'g', label='Regression Line')
plt.scatter(X_test, y_test, label='Raw Data')
plt.title('LR wrt Test Data')
plt.xlabel('Time Studied (in Hrs)')
plt.ylabel('Score (in %)')
plt.legend()
plt.show()

line = (LR.coef_ * X) + LR.intercept_
plt.scatter(X, y, label='Raw Data')
plt.plot(X, line, 'g', label='Regression Line')
plt.title('LR wrt to Whole Dataset')
plt.xlabel('Time Studied (in Hrs)')
plt.ylabel('Score (in %)')
plt.legend()
plt.show()

h = float(input('Hours : ')) # According to the question, i/p should be 9.25
h = np.reshape(h,(-1,1))
pred_q = LR.predict(h)
print(f'Predicted Score : {round(pred_q[0],2)}%')

from sklearn import metrics
print(f'Mean Absolute Error : {metrics.mean_absolute_error(y_test, pred)}')
