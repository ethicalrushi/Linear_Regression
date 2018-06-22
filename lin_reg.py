from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing datasets
features = datasets.load_diabetes()
df = pd.DataFrame(features.data[:,2])
df_target = pd.DataFrame(features.target)

X = np.array(df)
y = np.array(df_target)

X_train = X[:-50]
X_test = X[-50:]
y_train = y[:-50]
y_test = y[-50:]

#Linear Regressor using sklearn

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
error = mean_squared_error(y_pred, y_test)


#Linear Regressor From Scratch using gradient descent algorithm

def gradient(X_train, y_train, m , b, N):
    m_gradient = 0.0
    b_gradient = 0.0
    for i in range(N):
        y_current = y_train[i]
        x_current = X_train[i]
        m_gradient += -2*(y_current-(m *x_current + b))*(x_current)/N
        b_gradient +=-2*(y_current-(m *x_current + b))/N
    
  
    return [m_gradient, b_gradient]







no_iterations = 10000
learning_rate = 0.1
m =0
b= 0

N = len(X_train)
for i in range(no_iterations):
   
   [m_gradient, b_gradient]= gradient(X_train, y_train, m , b, N)
   b = b - (learning_rate * b_gradient)
   m = m - (learning_rate * m_gradient)
   
   
print(b, m)


   
#Comparing the sklearn(blue) and custom linear regressor(black) graphically   
plt.scatter(X_test, y_test, color='red')
plt.title('Linear Regression')
plt.xlabel('Age')
plt.ylabel('Sugar level')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.plot(X_test,( m*X_test)+b, color='black')
plt.show()

        
    
    
    
    
    





