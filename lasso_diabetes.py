#determining the baseline prediction of disease progression for future patients
import pandas as pd
import numpy as np 
from sklearn.linear_model import Lasso
from sklearn import linear_model,datasets 
from sklearn.model_selection import GridSearchCV

diabetes = datasets.load_diabetes()
X_train = diabetes.data[:310]
y_train = diabetes.data[:310]

X_test = diabetes.data[310:]
y_test = diabetes.data[310:]

lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)

estimator = GridSearchCV(lasso,dict(alpha= alphas))
estimator.fit(X_train,y_train)
#print(estimator.best_score_)
#print(estimator.best_estimator_)
x = estimator.predict(X_test)
print(x)

