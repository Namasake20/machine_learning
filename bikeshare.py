#developing model for predicting bike demand for a given date time
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sn
import numpy as np 
plt.style.use('seaborn')

# modeling utilities
import scipy.stats as stats
from sklearn import metrics
from sklearn import preprocessing
from sklearn import  linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV



hour_df = pd.read_csv('hour.csv')
#print("Shape of dataset::{}".format(hour_df.shape))
#print(hour_df.head())
#print(hour_df.dtypes)
hour_df.rename(columns={'instant':'rec_id','dteday':'datetime','holiday':'is_holiday','workingday':'is_workingday','weathersit':'weather_condition','hum':'humidity',
'mnth':'month','cnt':'total_count','hr':'hour','yr':'year'},inplace=True)

# date time conversion
hour_df['datetime'] = pd.to_datetime(hour_df.datetime)

# categorical variables
hour_df['season'] = hour_df.season.astype('category')
hour_df['is_holiday'] = hour_df.is_holiday.astype('category')
hour_df['weekday'] = hour_df.weekday.astype('category')
hour_df['weather_condition'] = hour_df.weather_condition.astype('category')
hour_df['is_workingday'] = hour_df.is_workingday.astype('category')
hour_df['month'] = hour_df.month.astype('category')
hour_df['year'] = hour_df.year.astype('category')
hour_df['hour'] = hour_df.hour.astype('category')
"""
fig,ax = plt.subplots()
sn.pointplot(data=hour_df[['hour','total_count','season']],x='hour',y='total_count',hue='season',ax=ax)
ax.set(title="Season wise hourly distribution of counts")

fig,ax = plt.subplots()
#sn.barplot(data=hour_df[['month','total_count']],x="month",y="total_count")
sn.violinplot(data=hour_df[['year','total_count']], x="year",y="total_count")
ax.set(title="Monthly distribution of counts")
plt.show()
"""

#outliers
#fig,(ax1,ax2)= plt.subplots(ncols=2)
#sn.boxplot(data=hour_df[['total_count','casual','registered']],ax=ax1)
#sn.boxplot(data=hour_df[['temp','windspeed']],ax=ax2)
#plt.show()

#correlation
#corrMatt = hour_df[["temp","atemp","humidity","windspeed","casual","registered","total_count"]].corr()
#mask = np.array(corrMatt)
#mask[np.tril_indices_from(mask)] = False
#sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
#plt.show()

#modeling
def fit_transform_ohe(df,col_name):
    """This function performs one hot encoding for the specified
        column.
    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        col_name: the column to be one hot encoded
    Returns:
        tuple: label_encoder, one_hot_encoder, transformed column as pandas Series
    """
    # label encode the column
    le = preprocessing.LabelEncoder()
    le_labels = le.fit_transform(df[col_name])
    df[col_name+'_label'] = le_labels
    # one hot encoding
    ohe = preprocessing.OneHotEncoder()
    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    return le,ohe,features_df
    
# given label encoder and one hot encoder objects, encode attribute to ohe
def transform_ohe(df,le,ohe,col_name):
    """This function performs one hot encoding for the specified
        column using the specified encoder objects.

    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        le(Label Encoder): the label encoder object used to fit label encoding
        ohe(One Hot Encoder): the onen hot encoder object used to fit one hot encoding
        col_name: the column to be one hot encoded

    Returns:
        tuple: transformed column as pandas Series

    """
    # label encode
    col_labels = le.transform(df[col_name])
    df[col_name+'_label'] = col_labels
    
    # ohe 
    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    
    return features_df
X, X_test, y, y_test = train_test_split(hour_df.iloc[:,0:-3], hour_df.iloc[:,-1],test_size=0.33, random_state=42)

X.reset_index(inplace=True)
y = y.reset_index()

X_test.reset_index(inplace=True)
y_test = y_test.reset_index()

#print("Training set::{}{}".format(X.shape,y.shape))
#print("Testing set::{}".format(X_test.shape))

#normality test
#stats.probplot(y.total_count.tolist(), dist="norm", plot=plt)
#plt.show()

cat_attr_list = ['season','is_holiday','weather_condition','is_workingday','hour','weekday','month','year']
numeric_feature_cols = ['temp','humidity','windspeed','hour','weekday','month','year']
subset_cat_features =  ['season','is_holiday','weather_condition','is_workingday']

encoded_attr_list = []
for col in cat_attr_list:
    return_obj = fit_transform_ohe(X,col)
    encoded_attr_list.append({'label_enc':return_obj[0],'ohe_enc':return_obj[1],'feature_df':return_obj[2],'col_name':col})

feature_df_list = [X[numeric_feature_cols]]
feature_df_list.extend([enc['feature_df'] \
                        for enc in encoded_attr_list \
                        if enc['col_name'] in subset_cat_features])

train_df_new = pd.concat(feature_df_list, axis=1)
#print("Shape::{}".format(train_df_new.shape))

X = train_df_new
y= y.total_count.values.reshape(-1,1)

lin_reg = linear_model.LinearRegression()
model1 = lin_reg.fit(X,y)

#k-fold crossvalidation
predicted = cross_val_predict(lin_reg, X, y, cv=10) #model object, predictors, and targets as inputs

#model evaluation showing scatter plot between residuals and observed values
"""
fig, ax = plt.subplots()
ax.scatter(y, y-predicted)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
plt.show()
r2_scores = cross_val_score(lin_reg, X, y, cv=10)
mse_scores = cross_val_score(lin_reg, X, y, cv=10,scoring='neg_mean_squared_error')
fig, ax = plt.subplots()
ax.plot([i for i in range(len(r2_scores))],r2_scores,lw=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('R-Squared')
ax.title.set_text("Cross Validation Scores, Avg:{}".format(np.average(r2_scores)))
plt.show()
"""

test_encoded_attr_list = []
for enc in encoded_attr_list:
    col_name = enc['col_name']
    le = enc['label_enc']
    ohe = enc['ohe_enc']
    test_encoded_attr_list.append({'feature_df':transform_ohe(X_test,le,ohe,col_name),'col_name':col_name})
    
test_feature_df_list = [X_test[numeric_feature_cols]]
test_feature_df_list.extend([enc['feature_df'] \
                             for enc in test_encoded_attr_list \
                             if enc['col_name'] in subset_cat_features])

test_df_new = pd.concat(test_feature_df_list, axis=1) 
#print("Shape::{}".format(test_df_new.shape))
#print(test_df_new.head())
X_test = test_df_new
y_test = y_test.total_count.values.reshape(-1,1)

y_pred = model1.predict(X_test)

residuals = y_test-y_pred

#MSE

r2_score = lin_reg.score(X_test,y_test)
print("R-squared::{}".format(r2_score))
print("MSE: %.2f"% metrics.mean_squared_error(y_test, y_pred))


#residual plot on the train dataset
fig, ax = plt.subplots()
ax.scatter(y_test, residuals)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residuals')
ax.title.set_text("Residual Plot with R-Squared={}".format(np.average(r2_score)))
plt.show()

"""
import statsmodels.api as sm

# Set the independent variable
X = X.values.tolist()

# This handles the intercept. 
# Statsmodel takes 0 intercept by default
X = sm.add_constant(X)

X_test = X_test.values.tolist()
X_test = sm.add_constant(X_test)


# Build OLS model
model = sm.OLS(y, X)
results = model.fit()

# Get the predicted values for dependent variable
pred_y = results.predict(X_test)

# View Model stats
print(results.summary())
"""

