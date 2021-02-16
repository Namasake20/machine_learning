from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster

np.set_printoptions(suppress=True)

#load data
data = load_breast_cancer()
X = data.data
y = data.target
Z = linkage(X, 'ward')
#print(Z)
plt.figure(figsize=(8, 3))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
dendrogram(Z)
plt.axhline(y=10000, c='k', ls='--', lw=0.5)
plt.show()
"""
max_dist = 10000
hc_labels = fcluster(Z, max_dist, criterion='distance')
"""
