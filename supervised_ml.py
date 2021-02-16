import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib
import os 

df = pd.DataFrame([{'Name':'Masake','OverallGrade':'A','Obedient':'Y','Researchscore':'90','ProjectScore':'78','Recommend':'Yes'},
{'Name':'Maggy','OverallGrade':'c','Obedient':'N','Researchscore':'70','ProjectScore':'58','Recommend':'Yes'},
{'Name':'Wangari','OverallGrade':'B','Obedient':'N','Researchscore':'40','ProjectScore':'30','Recommend':'No'},
{'Name':'Sascha','OverallGrade':'A','Obedient':'Y','Researchscore':'70','ProjectScore':'76','Recommend':'Yes'},
{'Name':'Brian','OverallGrade':'F','Obedient':'N','Researchscore':'30','ProjectScore':'30','Recommend':'No'},
{'Name':'Uhuru','OverallGrade':'D','Obedient':'N','Researchscore':'30','ProjectScore':'40','Recommend':'No'},
{'Name':'Winnie','OverallGrade':'A','Obedient':'Y','Researchscore':'90','ProjectScore':'90','Recommend':'Yes'}])

#print(df)
# get features and corresponding outcomes
feature_names = ['OverallGrade', 'Obedient', 'Researchscore','ProjectScore']
training_features = df[feature_names]

outcome_name = ['Recommend']
outcome_labels = df[outcome_name]
#print(training_features)
#print(outcome_labels)
# list down features based on type
numeric_feature_names = ['Researchscore', 'ProjectScore']
categoricial_feature_names = ['OverallGrade', 'Obedient']

ss = StandardScaler()
# fit scaler on numeric features
ss.fit(training_features[numeric_feature_names])
# scale numeric features now
training_features[numeric_feature_names] = ss.transform(training_features[numeric_feature_names])
#print(training_features)

training_features = pd.get_dummies(training_features,columns=categoricial_feature_names)
print(training_features)
# get list of new categorical features
categorical_engineered_features = list(set(training_features.columns) - set(numeric_feature_names))


# fit the model
lr = LogisticRegression()
model = lr.fit(training_features,np.array(outcome_labels['Recommend']))
#print(model)
#model evaluation
pred_labels = model.predict(training_features)
actual_labels = np.array(outcome_labels['Recommend'])

print('Accuracy:', float(accuracy_score(actual_labels,pred_labels))*100, '%')
print('Classification Stats:')
print(classification_report(actual_labels, pred_labels))
"""
# save models to be deployed on your server
if not os.path.exists('Model'):
    os.mkdir('Model')
if not os.path.exists('Scaler'):
    os.mkdir('Scaler')

joblib.dump(model, r'Model/model.pickle')
joblib.dump(ss, r'Scaler/scaler.pickle')
"""
# load model and scaler objects
model = joblib.load(r'Model/model.pickle')
scaler = joblib.load(r'Scaler/scaler.pickle')

# data retrieval
new_data = pd.DataFrame([{'Name': 'Nathan', 'OverallGrade': 'F','Obedient': 'N', 'Researchscore': 30, 'ProjectScore': 20},{'Name': 'Thomas', 'OverallGrade': 'A',
                   'Obedient': 'Y', 'Researchscore': 78, 'ProjectScore': 80}])
new_data = new_data[['Name', 'OverallGrade', 'Obedient','Researchscore', 'ProjectScore']]
#print(new_data)

# data preparation
prediction_features = new_data[feature_names]

# scaling
prediction_features[numeric_feature_names] = scaler.transform(prediction_features[numeric_feature_names])
# engineering categorical variables
prediction_features = pd.get_dummies(prediction_features,columns=categoricial_feature_names)
#print(prediction_features)
# add missing categorical feature columns
current_categorical_engineered_features = set(prediction_features.columns) - set(numeric_feature_names)
missing_features = set(categorical_engineered_features) - current_categorical_engineered_features

for feature in missing_features:
    # add zeros since feature is absent in these data samples
    prediction_features[feature] = [0] * len(prediction_features)

#print(prediction_features)
# predict using model
predictions = model.predict(prediction_features)
new_data['Recommend'] = predictions
print(new_data)
