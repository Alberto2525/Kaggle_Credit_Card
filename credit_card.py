#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Importing the dataset
df_dataset = pd.read_csv('Credit Card.csv')

#Data Exploration
df_dataset.shape
description = df_dataset.describe()
df_dataset.isnull().sum()

#Dealing with the nan values
sns.distplot(df_dataset['MINIMUM_PAYMENTS'].dropna(),kde = False)
sns.scatterplot(x = range(0,len(df_dataset['MINIMUM_PAYMENTS'])),y = df_dataset['MINIMUM_PAYMENTS'])
df_dataset['MINIMUM_PAYMENTS'].fillna(value = df_dataset['MINIMUM_PAYMENTS'].median(),inplace = True)

sns.distplot(df_dataset['CREDIT_LIMIT'],kde = False)
sns.scatterplot(x = range(0,len(df_dataset['CREDIT_LIMIT'])),y = df_dataset['CREDIT_LIMIT'])
df_dataset.dropna(inplace = True)
df_dataset.isna().sum()

#Some data visualization
sns.jointplot(x = df_dataset['BALANCE'],y = df_dataset['CREDIT_LIMIT'])
sns.jointplot(x = df_dataset['PAYMENTS'],y = df_dataset['BALANCE'])
sns.jointplot(x = df_dataset['TENURE'],y = df_dataset['CREDIT_LIMIT'])

#Droping the CUST_ID since is not needed and creating my x variable
x = df_dataset.drop(columns = 'CUST_ID')
x.shape

#Scaling the features
scaler = StandardScaler()
x = scaler.fit_transform(x)

#Creating the model, first using the elbow method for the right clusters
wcss = []
for i in range(1,34):
     k_means = KMeans(n_clusters = i)
     k_means.fit(x)
     wcss.append(k_means.inertia_)
     
#Plotting the elbow method
plt.plot(range(1,34),wcss)
plt.title('Elbow Method')
plt.xlabel('N_Clusters')
plt.ylabel('WCSS values')
plt.show()

#Applying k-means with the right number of clusters
k_means = KMeans(n_clusters = 10)
y_means = k_means.fit_predict(x)

#TODO
#making sense out of the clusters, need to learn PCA to reduce dimension for visualization