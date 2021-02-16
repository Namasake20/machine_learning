import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import random
plt.style.use('seaborn')

def f(m,X,c):
    """Linear regression"""
    return [m*x + c for x in X]

X = [i for i in range(10)]
y = [x + random.random() for x in X]

m,c = 1,0
y_hat = f(m,X,c)


#no. of iterations,weight,bias,learning rate
e,m,c,a = 10,0,0,0.01

for i in range(e):
    y_hat = f(m,X,c)
    n = len(y_hat)

    #compute partial derivative of MSE w.r.t m and c

    dm = (-2/n) * sum([X[i] * (y[i] - y_hat[i]) for i in range(n)])
    dc = (-2/n) * sum([X[i] - y_hat[i] for i in range(n)])

    #update weight and bias 
    m = m -a * dm
    c = c - a * dc 
    
    print(f"Epoch {i}: m = {m:.2f}, c={c:.2f}")

y_hat = f(m,X,c)
plt.plot(X,y,'.',c='r')
plt.plot(X,y_hat)
plt.show()

