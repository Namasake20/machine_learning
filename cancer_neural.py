from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np 
from sklearn import metrics


cancer = load_breast_cancer()

X_train = cancer.data[:340]
y_train = cancer.target[:340]

X_test = cancer.data[340:]
y_test = cancer.target[340:]

#model definition
model = Sequential()
model.add(Dense(15, input_dim=30, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #sigmoid activation converts a real valued input into binary output(0,1)

#model compilation
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train,y_train,epochs=20,batch_size=50) #epochs parameter indicates one complete forward and backward pass of all the training examples
predictions = model.predict_classes(X_test)
print('Accuracy:', metrics.accuracy_score(y_true=y_test, y_pred=predictions))
print(metrics.classification_report(y_true=y_test, y_pred=predictions))

