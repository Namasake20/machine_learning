import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn import linear_model


digits  = datasets.load_digits()

plt.figure(figsize=(3,3))
plt.imshow(digits.images[10], cmap=plt.cm.gray_r)

#actual image pixel matrix
digits.images[10]

# flattened vector
digits.data[10]

#image class label
print(digits.target[10])

X_digits = digits.data
y_digits = digits.target

num_data_points = len(X_digits)
X_train = X_digits[:int(.7 * num_data_points)]
y_train = y_digits[:int(.7 * num_data_points)]

X_test = X_digits[int(.7 * num_data_points):]
y_test = y_digits[int(.7 * num_data_points):]
#print(X_train.shape, X_test.shape)

logistic = linear_model.LogisticRegression()
logistic.fit(X_train, y_train)
print('Logistic Regression mean accuracy: %f' % logistic.score(X_test, y_test))

