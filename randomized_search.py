import scipy
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#load data
data = load_breast_cancer()
X = data.data
y = data.target

# prepare datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {'C': scipy.stats.expon(scale=10),'gamma': scipy.stats.expon(scale=.1),'kernel': ['rbf', 'linear']}
random_search = RandomizedSearchCV(SVC(random_state=42), param_distributions=param_grid,n_iter=50, cv=5)
random_search.fit(X_train, y_train)
print("Best parameters set found on development set:")
print(random_search.best_params_)
# get best model, predict and evaluate performance
rs_best = random_search.best_estimator_
rs_y_pred = rs_best.predict(X_test)
#meu.get_metrics(true_labels=y_test, predicted_labels=rs_y_pred)

