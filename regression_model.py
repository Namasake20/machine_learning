from sklearn import datasets
from sklearn.linear_model import Lasso
import numpy as np 
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV


diabetes = datasets.load_diabetes()
y = diabetes.target
X = diabetes.data
#print(X[:5])
feature_names=['age', 'sex', 'bmi', 'bp','s1', 's2', 's3', 's4', 's5', 's6']

X_train = diabetes.data[:310]
y_train = diabetes.target[:310]

X_test = diabetes.data[310:]
y_test = diabetes.data[310:]

lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)

estimator = GridSearchCV(lasso, dict(alpha=alphas))
estimator.fit(X_train, y_train)
estimator.predict(X_test)
print(estimator.predict(X_test))
