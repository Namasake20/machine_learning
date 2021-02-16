import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import seaborn as sn 
import numpy as np 
from scipy import stats
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#from skater.core.explanations import Interpretation
#from skater.model import InMemoryModel

white_wine = pd.read_csv('winequality-white.csv', sep=';')
red_wine = pd.read_csv('winequality-red.csv', sep=';')

# store wine type as an attribute
red_wine['wine_type'] = 'red'   
white_wine['wine_type'] = 'white'

# bucket wine quality scores into qualitative quality labels
red_wine['quality_label'] = red_wine['quality'].apply(lambda value: 'low'
                                                    if value <= 5 else 'medium'
                                                        if value <= 7 else 'high')
red_wine['quality_label'] = pd.Categorical(red_wine['quality_label'],categories=['low', 'medium', 'high'])
white_wine['quality_label'] = white_wine['quality'].apply(lambda value: 'low'
                                                    if value <= 5 else 'medium'
                                                       if value <= 7 else 'high')

white_wine['quality_label'] = pd.Categorical(white_wine['quality_label'],categories=['low', 'medium', 'high'])

# merge red and white wine datasets
wines = pd.concat([red_wine, white_wine])

# re-shuffle records just to randomize data points
wines = wines.sample(frac=1, random_state=42).reset_index(drop=True)
#print(white_wine.shape, red_wine.shape)
#print(wines.info())
print(wines.head())

#descriptive statistics
subset_attributes = ['residual sugar', 'total sulfur dioxide', 'sulphates','alcohol', 'volatile acidity', 'quality']
rs = round(red_wine[subset_attributes].describe(),2)
ws = round(white_wine[subset_attributes].describe(),2)
pd.concat([rs, ws], axis=1, keys=['Red Wine Statistics', 'White Wine Statistics'])

subset_attributes = ['alcohol', 'volatile acidity', 'pH', 'quality']
ls = round(wines[wines['quality_label'] == 'low'][subset_attributes].describe(),2)
ms = round(wines[wines['quality_label'] == 'medium'][subset_attributes].describe(),2)
hs = round(wines[wines['quality_label'] == 'high'][subset_attributes].describe(),2)
pd.concat([ls, ms, hs], axis=1, keys=['Low Quality Wine', 'Medium Quality Wine','High Quality Wine'])

#inferential statistics
F, p = stats.f_oneway(wines[wines['quality_label'] == 'low']['alcohol'],wines[wines['quality_label'] == 'medium']['alcohol'],wines[wines['quality_label'] == 'high']['alcohol'])
#print('ANOVA test for mean alcohol levels across wine samples with different quality ratings')
#print('F Statistic:', F, '\tp-value:', p)
F, p = stats.f_oneway(wines[wines['quality_label'] == 'low']['pH'],wines[wines['quality_label'] == 'medium']['pH'],wines[wines['quality_label'] == 'high']['pH'])
#print('\nANOVA test for mean pH levels across wine samples with different quality ratings')
#print('F Statistic:', F, '\tp-value:', p)
"""
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
f.suptitle('Wine Quality - Alcohol Content/pH', fontsize=14)
f.subplots_adjust(top=0.85, wspace=0.3)
sn.boxplot(x="quality_label", y="alcohol",data=wines, ax=ax1)
ax1.set_xlabel("Wine Quality Class",size = 12,alpha=0.8)
ax1.set_ylabel("Wine Alcohol %",size = 12,alpha=0.8)
sn.boxplot(x="quality_label", y="pH", data=wines, ax=ax2)
ax2.set_xlabel("Wine Quality Class",size = 12,alpha=0.8)
ax2.set_ylabel("Wine pH",size = 12,alpha=0.8)
plt.show()
"""

#univariate analysis
"""
red_wine.hist(bins=15, color='red', edgecolor='black', linewidth=1.0,
              xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout(rect=(0, 0, 1.2, 1.2))   
rt = plt.suptitle('Red Wine Univariate Plots', x=0.65, y=1.25, fontsize=14)  
white_wine.hist(bins=15, color='white', edgecolor='black', linewidth=1.0,
              xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout(rect=(0, 0, 1.2, 1.2))   
wt = plt.suptitle('White Wine Univariate Plots', x=0.65, y=1.25, fontsize=14)
plt.show()
"""
'''
fig = plt.figure(figsize = (10,4))
title = fig.suptitle("Residual Sugar Content in Wine", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax1 = fig.add_subplot(1,2, 1)
ax1.set_title("Red Wine")
ax1.set_xlabel("Residual Sugar")
ax1.set_ylabel("Frequency")
ax1.set_ylim([0, 2500])
ax1.text(8, 1000, r'$\mu$='+str(round(red_wine['residual sugar'].mean(),2)),fontsize=12)
r_freq, r_bins, r_patches = ax1.hist(red_wine['residual sugar'], color='red', bins=15,edgecolor='black', linewidth=1)
ax2 = fig.add_subplot(1,2, 2)
ax2.set_title("White Wine")
ax2.set_xlabel("Residual Sugar")
ax2.set_ylabel("Frequency")
ax2.set_ylim([0, 2500])
ax2.text(30, 1000, r'$\mu$='+str(round(white_wine['residual sugar'].mean(),2)),fontsize=12)
w_freq, w_bins, w_patches = ax2.hist(white_wine['residual sugar'], color='white', bins=15, edgecolor='black', linewidth=1)

plt.show()
'''
#multivariate analysis
'''
f, ax = plt.subplots(figsize=(10, 5))
corr = wines.corr()
hm = sn.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=12)
plt.show()

cols = ['wine_type', 'quality', 'sulphates', 'volatile acidity']
pp = sn.pairplot(wines[cols], hue='wine_type', height=1.8, aspect=1.8,palette={"red": "#FF9999", "white": "#FFE888"},plot_kws=dict(edgecolor="black", linewidth=0.5))
fig = pp.fig
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14)
plt.show()


rj = sn.jointplot(x='quality', y='sulphates', data=red_wine,kind='reg', ylim=(0, 2),color='red', space=0, size=4.5, ratio=4)
rj.ax_joint.set_xticks(list(range(3,9)))
fig = rj.fig
fig.subplots_adjust(top=0.9)
t = fig.suptitle('Red Wine Sulphates - Quality', fontsize=12)
wj = sn.jointplot(x='quality', y='sulphates', data=white_wine,kind='reg', ylim=(0, 2),color='#FFE160', space=0, size=4.5, ratio=4)
wj.ax_joint.set_xticks(list(range(3,10)))
fig = wj.fig
fig.subplots_adjust(top=0.9)
t = fig.suptitle('White Wine Sulphates - Quality', fontsize=12)
plt.show()

g = sn.FacetGrid(wines, col="wine_type", hue='quality_label',col_order=['red', 'white'], hue_order=['low', 'medium', 'high'],aspect=1.2, size=3.5, palette=sn.light_palette('navy', 3))
g.map(plt.scatter, "volatile acidity", "alcohol", alpha=0.9,edgecolor='white', linewidth=0.5)
fig = g.fig
fig.subplots_adjust(top=0.8, wspace=0.3)
fig.suptitle('Wine Type - Alcohol - Quality - Acidity', fontsize=14)
l = g.add_legend(title='Wine Quality Class')
plt.show()


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
f.suptitle('Wine Type - Quality - Acidity', fontsize=14)
sn.violinplot(x="quality", y="volatile acidity", hue="wine_type",data=wines, split=True, inner="quart", linewidth=1.3,palette={"red": "#FF9999", "white": "white"}, ax=ax1)
ax1.set_xlabel("Wine Quality",size = 12,alpha=0.8)
ax1.set_ylabel("Wine Fixed Acidity",size = 12,alpha=0.8)
sn.violinplot(x="quality_label", y="volatile acidity", hue="wine_type",data=wines, split=True, inner="quart", linewidth=1.3,palette={"red": "#FF9999", "white": "white"}, ax=ax2)
ax2.set_xlabel("Wine Quality Class",size = 12,alpha=0.8)
ax2.set_ylabel("Wine Fixed Acidity",size = 12,alpha=0.8)
l = plt.legend(loc='upper right', title='Wine Type')
plt.show()
'''

#predicting wine types
wtp_features = wines.iloc[:,:-3]
wtp_feature_names = wtp_features.columns
wtp_class_labels = np.array(wines['wine_type'])
wtp_train_X, wtp_test_X, wtp_train_y, wtp_test_y = train_test_split(wtp_features,wtp_class_labels, test_size=0.3, random_state=42)
#print(Counter(wtp_train_y), Counter(wtp_test_y))
#print('Features:', list(wtp_feature_names))

# Define the scaler
wtp_ss = StandardScaler().fit(wtp_train_X)

# Scale the train set
wtp_train_SX = wtp_ss.transform(wtp_train_X)

# Scale the test set
wtp_test_SX = wtp_ss.transform(wtp_test_X)

#model training
wtp_lr = LogisticRegression()
wtp_lr.fit(wtp_train_SX, wtp_train_y)

#predicting wine types
wtp_lr_predictions = wtp_lr.predict(wtp_test_SX)
print(wtp_lr_predictions)
acc = wtp_lr.score(wtp_train_SX,wtp_train_y)
print(acc)
#Deep learning
le = LabelEncoder()
le.fit(wtp_train_y)

# encode wine type labels
wtp_train_ey = le.transform(wtp_train_y)
wtp_test_ey = le.transform(wtp_test_y)

#dnn
wtp_dnn_model = Sequential()
wtp_dnn_model.add(Dense(16, activation='relu', input_shape=(11,)))
wtp_dnn_model.add(Dense(16, activation='relu'))
wtp_dnn_model.add(Dense(16, activation='relu'))
wtp_dnn_model.add(Dense(1, activation='sigmoid'))

wtp_dnn_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

history = wtp_dnn_model.fit(wtp_train_SX, wtp_train_ey, epochs=10, batch_size=5,shuffle=True, validation_split=0.1, verbose=1)

#prediction
wtp_dnn_ypred = wtp_dnn_model.predict_classes(wtp_test_SX)
wtp_dnn_predictions = le.inverse_transform(wtp_dnn_ypred)
print(wtp_dnn_predictions)
print(metrics.accuracy_score(y_true=wtp_test_y,y_pred=wtp_dnn_predictions))
print(metrics.classification_report(y_true=wtp_test_y,y_pred=wtp_dnn_predictions))


#interpretation
#wtp_interpreter = Interpretation(wtp_test_SX, feature_names=wtp_features.columns)
#wtp_im_model = InMemoryModel(wtp_lr.predict_proba, examples=wtp_train_SX,target_names=wtp_lr.classes_)
#plots = wtp_interpreter.feature_importance.plot_feature_importance(wtp_im_model,ascending=False)

#wine quality prediction
wqp_features = wines.iloc[:,:-3]
wqp_class_labels = np.array(wines['quality_label'])
wqp_label_names = ['low', 'medium', 'high']
wqp_feature_names = list(wqp_features.columns)
wqp_train_X, wqp_test_X, wqp_train_y, wqp_test_y = train_test_split(wqp_features,wqp_class_labels, test_size=0.3, random_state=42)

#print(Counter(wqp_train_y), Counter(wqp_test_y))
#print('Features:', wqp_feature_names)

# Define the scaler
wqp_ss = StandardScaler().fit(wqp_train_X)

# Scale the train set
wqp_train_SX = wqp_ss.transform(wqp_train_X)

# Scale the test set
wqp_test_SX = wqp_ss.transform(wqp_test_X)

# train the model
wqp_dt = DecisionTreeClassifier()
wqp_dt.fit(wqp_train_SX, wqp_train_y)
# predict and evaluate performance
wqp_dt_predictions = wqp_dt.predict(wqp_test_SX)
#print(wqp_dt_predictions)

wqp_dt_feature_importances = wqp_dt.feature_importances_
wqp_dt_feature_names, wqp_dt_feature_scores = zip(*sorted(zip(wqp_feature_names,wqp_dt_feature_importances), key=lambda x: x[1]))
y_position = list(range(len(wqp_dt_feature_names)))
plt.barh(y_position, wqp_dt_feature_scores, height=0.6, align='center')
plt.yticks(y_position , wqp_dt_feature_names)
plt.xlabel('Relative Importance Score')
plt.ylabel('Feature')
t = plt.title('Feature Importances for Decision Tree')
#plt.show()

# train the model
wqp_rf = RandomForestClassifier()
wqp_rf.fit(wqp_train_SX, wqp_train_y)

# predict and evaluate performance
wqp_rf_predictions = wqp_rf.predict(wqp_test_SX)

#model tuning
#print(wqp_rf.get_params())
param_grid = {
    'n_estimators': [100, 200, 300, 500],'max_features': ['auto', None, 'log2']              
}
wqp_clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5,scoring='accuracy')
wqp_clf.fit(wqp_train_SX, wqp_train_y)
#print(wqp_clf.best_params_)
'''
results = wqp_clf.cv_results_
for param, score_mean, score_sd in zip(results['params'], results['mean_test_score'],results['std_test_score']):
    print(param, round(score_mean, 4), round(score_sd, 4))
'''
#train a new random forest model with the tuned hyperparameters and evaluate its performance on the test data.
wqp_rf = RandomForestClassifier(n_estimators=200, max_features='auto', random_state=42)
wqp_rf.fit(wqp_train_SX, wqp_train_y)
wqp_rf_predictions = wqp_rf.predict(wqp_test_SX)
#print(wqp_rf_predictions)



