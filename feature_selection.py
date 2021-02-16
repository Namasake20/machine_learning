import pandas as pd 
import numpy as np 
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

np.set_printoptions(suppress=True)
pt = np.get_printoptions()['threshold']

from sklearn.datasets import load_breast_cancer
bc_data = load_breast_cancer()
bc_features = pd.DataFrame(bc_data.data, columns=bc_data.feature_names)
bc_classes = pd.DataFrame(bc_data.target, columns=['IsMalignant'])

# build featureset and response class labels
bc_X = np.array(bc_features)
bc_y = np.array(bc_classes).T[0]
#print('Feature set shape:', bc_X.shape)
#print('Response class shape:', bc_y.shape)
np.set_printoptions(threshold=30)
"""
print('Feature set data [shape: '+str(bc_X.shape)+']')
print(np.round(bc_X, 2), '\n')
print('Feature names:')
print(np.array(bc_features.columns), '\n')
print('Response Class label data [shape: '+str(bc_y.shape)+']')
print(bc_y, '\n')
print('Response variable name:', np.array(bc_classes.columns))
"""
np.set_printoptions(threshold=pt)

skb = SelectKBest(score_func=chi2, k=15)
skb.fit(bc_X, bc_y)
feature_scores = [(item, score) for item, score in zip(bc_data.feature_names,skb.scores_)]
#print(sorted(feature_scores, key=lambda x: -x[1])[:10])

select_features_kbest = skb.get_support()
feature_names_kbest = bc_data.feature_names[select_features_kbest]
feature_subset_df = bc_features[feature_names_kbest]
bc_SX = np.array(feature_subset_df)
'''
print(bc_SX.shape)
print(feature_names_kbest)
print(np.round(feature_subset_df.iloc[20:25], 2))

'''
# build logistic regression model
lr = LogisticRegression()

# evaluating accuracy for model built on full featureset
full_feat_acc = np.average(cross_val_score(lr, bc_X, bc_y, scoring='accuracy', cv=5))

# evaluating accuracy for model built on selected featureset
sel_feat_acc = np.average(cross_val_score(lr, bc_SX, bc_y, scoring='accuracy', cv=5))
print('Model accuracy statistics with 5-fold cross validation')
print('Model accuracy with complete feature set', bc_X.shape, ':', full_feat_acc)
print('Model accuracy with selected feature set', bc_SX.shape, ':', sel_feat_acc)

#principal component annalysis
pca = PCA(n_components=3)
pca.fit(bc_X)
#print(pca.explained_variance_ratio_)
bc_pca = pca.transform(bc_X)
pca_acc = np.average(cross_val_score(lr, bc_pca, bc_y, scoring='accuracy', cv=5))
print("Model accuracy using principal accuracy analysis", bc_pca.shape,':',pca_acc)

