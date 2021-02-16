import numpy as np 
import math
import pandas as pd 
import matplotlib.pyplot as plt 
import datetime
import matplotlib.mlab as mlab

cs_df = pd.read_excel(io=r'Online Retail.xlsx')
#print(cs_df.head())
cs_df.Country.value_counts().reset_index().head(n=10)
cs_df.CustomerID.unique().shape
#print(cs_df.CustomerID.unique().shape)

(cs_df.CustomerID.value_counts()/sum(cs_df.CustomerID.value_counts())*100).head(n=13).cumsum()
cs_df.StockCode.unique().shape #unique  items 
cs_df.Description.unique().shape #unique descriptions
cat_des_df = cs_df.groupby(["StockCode","Description"]).count().reset_index()
cat_des_df.StockCode.value_counts()[cat_des_df.StockCode.value_counts()>1].reset_index().head()

cs_df[cs_df['StockCode'] == cat_des_df.StockCode.value_counts()[cat_des_df.StockCode.value_counts()>1].reset_index()['index'][6]]['Description'].unique()
cs_df.Quantity.describe()
#quantity = cs_df.Quantity.describe()
#print(quantity)
cs_df.UnitPrice.describe()

# Separate data for one geography
cs_df = cs_df[cs_df.Country == 'United Kingdom']

# Separate attribute for total amount
cs_df['amount'] = cs_df.Quantity*cs_df.UnitPrice

# Remove negative or return transactions
cs_df = cs_df[~(cs_df.amount<0)]
#print(cs_df.head())
cs_df = cs_df[~(cs_df.CustomerID.isnull())]

#Recency
refrence_date = cs_df.InvoiceDate.max()
refrence_date = refrence_date + datetime.timedelta(days = 1)
#print(refrence_date)
cs_df['days_since_last_purchase'] = refrence_date - cs_df.InvoiceDate
cs_df['days_since_last_purchase_num'] = cs_df['days_since_last_purchase'].astype('timedelta64[D]')
customer_history_df = cs_df.groupby("CustomerID").min().reset_index()[['CustomerID', 'days_since_last_purchase_num']]

customer_history_df.rename(columns={'days_since_last_purchase_num':'recency'},inplace=True)
x = customer_history_df.recency
mu = np.mean(customer_history_df.recency)
sigma = math.sqrt(np.var(customer_history_df.recency))
n, bins, patches = plt.hist(x, 1000, facecolor='green', alpha=0.75)
# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
plt.xlabel('Recency in days')
plt.ylabel('Number of transactions')
plt.title(r'$\mathrm{Histogram\ of\ sales\ recency}\ $')
plt.grid(True)
plt.show()
