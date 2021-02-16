import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('payment_fraud.csv')
df.head()
print(df.head(10))
print(df['paymentMethod'].value_counts())
"""
# Split dataset up into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'],test_size=0.33, random_state=17)

clf = LogisticRegression().fit(X_train, y_train)

# Make predictions on test set
y_pred = clf.predict(X_test)
print(accuracy_score(y_pred, y_test))

#evaluation
# Compare test set predictions with ground truth labels
#print(confusion_matrix(y_test, y_pred))
"""