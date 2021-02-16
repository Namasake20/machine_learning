import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
np.set_printoptions(suppress=True)

views = pd.DataFrame([1295., 25., 19000., 5., 1., 300.], columns=['views'])
#print(views)

#standardized scaling
ss = StandardScaler()
views['zscore'] = ss.fit_transform(views[['views']])


#min-max scaling
mms = MinMaxScaler()
views['minmax'] = mms.fit_transform(views[['views']])

#robust-scaling
rs = RobustScaler()
views['robust'] = rs.fit_transform(views[['views']])
print(views)