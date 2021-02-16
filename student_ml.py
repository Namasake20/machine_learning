import pandas as pd 
import numpy as np 
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 
import pickle
from matplotlib import style


data = pd.read_csv("student-mat.csv",sep=';')

data = data[["G1","G2","G3","studytime","failures","absences"]]
#print(data.head())

predict = "G3"

X = np.array(data.drop([predict],1))
y = np.array(data[predict])


x_train,x_test,y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
"""
best = 0
for _ in range(50):
    x_train,x_test,y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    print(acc)
    
    #saving the model
    if acc > best:
        best = acc
        with open("studentModel.pickle", "wb") as f:
            pickle.dump(linear, f)
"""

pickle_in = open("studentModel.pickle","rb")
linear = pickle.load(pickle_in)


print("coefficent: ", linear.coef_) #m coefficient in y = mx + c
print("Intercept: ",linear.intercept_) # y-intercept,c
print("\n")

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x]) #displays -> predicted value(grade)[first grade,second grade,studytime,failures,absences]actual grade

#ploting data
style.use("ggplot")
p = "G2"
plt.scatter(data[p],data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()


