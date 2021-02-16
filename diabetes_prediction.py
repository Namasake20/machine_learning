import pandas as pd 
import numpy as np 
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model,preprocessing

data = pd.read_csv("diabetes_data_upload.csv")
#print(data.head())

le = preprocessing.LabelEncoder()


Age = data["Age"]
#Age = le.fit_transform(list(data["Age"])) -> not necessary since age is an integer already
Gender = le.fit_transform(list(data["Gender"]))
Polyuria = le.fit_transform(list(data["Polyuria"]))
Polydipsia = le.fit_transform(list(data["Polydipsia"]))
sudden_weight_loss = le.fit_transform(list(data["sudden weight loss"]))
weakness = le.fit_transform(list(data["weakness"]))
Polyphagia = le.fit_transform(list(data["Polyphagia"]))
Genital_thrush = le.fit_transform(list(data["Genital thrush"]))
visual_blurring = le.fit_transform(list(data["visual blurring"]))
Itching = le.fit_transform(list(data["Itching"]))
Irritability = le.fit_transform(list(data["Irritability"]))
delayed_healing = le.fit_transform(list(data["delayed healing"]))
partial_paresis = le.fit_transform(list(data["partial paresis"]))
muscle_stiffness = le.fit_transform(list(data["muscle stiffness"]))
Alopecia = le.fit_transform(list(data["Alopecia"]))
Obesity = le.fit_transform(list(data["Obesity"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(Age,Gender,Polyuria,Polydipsia,sudden_weight_loss,weakness,Polyphagia,Genital_thrush,visual_blurring,Itching,Irritability,
delayed_healing,partial_paresis,muscle_stiffness,Alopecia,Obesity,cls))

y = list(cls)

x_train, x_test, y_train ,y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
#print(x_train,y_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(acc)

predicted = model.predict(x_test)

for x in range(len(predicted)):
    print("Predicted: ",predicted[x],"Actual: ",y_test[x])
