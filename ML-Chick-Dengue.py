
# coding: utf-8

# In[79]:


import pandas as pd
import unicodecsv
import numpy as np 
from pandas import Series,DataFrame
from scipy.stats import norm
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().magic('matplotlib inline')

# For evaluating our ML results
from sklearn import metrics


# In[80]:

_data = 'C://Shesha//Data/problem_dataset.csv'
data_df = pd.read_csv(_data)
train = data_df
data_df.tail()


# In[81]:

data_df.shape


# In[82]:

## This is to find which are catogeries
data_df.info()


# In[83]:

list(data_df.columns)


# In[84]:

data_df['STATUS'].head()


# In[85]:

data_df.apply(lambda x: len(x.unique()))


# In[86]:

def mappingToCode(oldColName,newColName):
        
    '''ill_source = data_df['infection_source'].unique()
    ill_source_mapping = dict(zip(ill_source,range(0, len(ill_source) + 1)))
    data_df['infection_source_ID'] = data_df['infection_source'] \
                                   .map(ill_source_mapping) \
                                   .astype(int)'''
    ill_source = data_df[oldColName].unique()
    print(ill_source)
    ill_source_mapping = dict(zip(ill_source,range(0, len(ill_source) + 1)))
    data_df[newColName] = data_df[oldColName] \
                                   .map(ill_source_mapping) \
                                   .astype(int)
    return
            
            


# In[87]:

mappingToCode('infection_source','infection_source_ID')
mappingToCode('VECTOR','VECTOR_ID')
mappingToCode('COUNTRY_ID','COUNTRYID')
data_df.head()


# In[ ]:




# In[88]:

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# In[89]:

def normal(df):
    result = df.copy()
    for cols in df.columns:
        max_value = df[cols].max()
        min_value = df[cols].min()
        result[cols] = (df[cols] - min_value) / (max_value - min_value)
        


# In[90]:

## Check if null or missing values
data_df.apply(lambda x: sum(x.isnull()))


# In[91]:

''' data_df_vec1 = data_df[data_df['VECTOR'] == 'Aedes aegypti']


data_df_vec2 = data_df[data_df['VECTOR'] == 'Aedes albopictus']

data_df['VECTOR'].loc[data_df['VECTOR'] == 'Aedes albopictus'] = 1
data_df['VECTOR'].loc[data_df['VECTOR'] == 'Aedes aegypti'] = 0'''
        
ax = data_df.VECTOR_ID.value_counts().plot(kind='bar',title ="Dengue Vs Chikungunya",colormap='Paired')
ax.set_ylabel("Total cases ", fontsize=12)
ax.set_xlabel("0-Dengue 1-Chikungunya", fontsize=12)
plt.show()


# In[92]:

data_df_Y = data_df[data_df['YEAR'].isnull() == False]
len(data_df_Y['COUNTRY'].unique())


# In[93]:

NotyearDef = data_df[data_df['YEAR'].isnull() == True]
len(NotyearDef['COUNTRY'].unique())


# In[94]:

## Year wise w.r.t Source of infection
dfgrp = data_df_Y.groupby(['YEAR','infection_source'])
newDf = dfgrp.size().unstack()
newDf.fillna(0,inplace=True)
newDf.head()
#newDf.plot.scatter()
newDf.plot(subplots=True, layout=(2, 3), figsize=(10, 10), sharex=False);

# dfgrp.plot.scatter(x='X',y='Y',c='m')


# In[95]:

## Year wise w.r.t Vector
dfgrp = data_df_Y.groupby(['YEAR','VECTOR'])[['infection_time']]
inf_df = dfgrp.mean().unstack()
inf_df.fillna(0,inplace=True)
ax = inf_df.plot()
ax.set_ylabel("Average infection time ", fontsize=12)
ax.set_xlabel("Vector types", fontsize=12)
plt.show()



# In[96]:

## Year wise w.r.t Vector
dfgrp = data_df_Y.groupby(['YEAR','COUNTRY_ID'])
newDf = dfgrp.size().unstack()
newDf.fillna(0,inplace=True)
newDf['CHN'].tail()


# In[97]:

## Year wise w.r.t Vector
dfgrp = data_df_Y.groupby(['YEAR','VECTOR'])
newDf = dfgrp.size().unstack()
newDf.fillna(0,inplace=True)
newDf.head()
#newDf.plot.scatter()
#newDf.plot(subplots=True, layout=(2, 3), figsize=(10, 10), sharex=False);


# In[118]:

## Selecting the columns required related to segmenting in to cluster for K-means
clusters = data_df_Y[data_df_Y.infection_time < 10]
#clusters_df = clusters[['X','Y','YEAR','infection_time','infection_source_ID','COUNTRYID']]
clusters_df = clusters[['YEAR','infection_time','infection_source_ID']]
X = clusters_df.iloc[:, ].values


# In[119]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()



# In[123]:

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
X[1]


# In[126]:

clusters_df =clusters[['X','Y','YEAR','infection_time','infection_source_ID','COUNTRYID']]
X = clusters_df.iloc[:, ].values
y_kmeans


# In[145]:


#y_kmeans -- Getting the cluster

plt.figure(figsize=(10, 10))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 10, c = 'red', label = 'Risk Region :1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 10, c = 'blue', label = 'Risk Region :2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 10, c = 'green', label = 'Risk Region :3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 10, c = 'Magenta', label = 'Risk Region :4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow')
plt.title('Clusters of Countries')
plt.xlabel('Infection Time (1..10)')
plt.ylabel('Country (1-100)')
plt.legend()
plt.show()


# In[128]:

#dfgrp = data_df_Y.groupby(['YEAR','infection_source']).aggregate(np.sum)
plt.figure()
#df.groupby(['Year','Fungicide']).sum().unstack().plot()
#data_df_Y.groupby(['YEAR','infection_source']).aggregate(np.sum).unstack()
#data_df_Y.groupby(['YEAR','infection_source']).sum().unstack()
dfgrp = data_df_Y.groupby(['YEAR','infection_source'])['COUNTRY'].value_counts()
dfgrp.plot(subplots=True, figsize=(6, 6),color='r');




# In[129]:

## How vectors are effecting the Regions

scat_plot_df = data_df[['X','Y','VECTOR_ID']]
Vec1_df = scat_plot_df[scat_plot_df['VECTOR_ID'] == 0]
Vec2_df = scat_plot_df[scat_plot_df['VECTOR_ID'] == 1]
ax = Vec1_df.plot.scatter(x='X',y='Y',c='r',label = 'Dengue')
Vec2_df.plot.scatter(x='X',y='Y',c='b',label = 'Chikungunya',figsize=(10, 10),ax=ax)
                       
                       


# In[130]:

scat_plot_df = data_df[['X','Y','infection_source_ID']]
plt.figure(figsize=(15, 15))
air_df = scat_plot_df[scat_plot_df['infection_source_ID'] == 0]
mos_df = scat_plot_df[scat_plot_df['infection_source_ID'] == 1]
wat_df = scat_plot_df[scat_plot_df['infection_source_ID'] == 2]
ax=air_df.plot.scatter(x='X',y='Y',c='r',label = 'Air Source')
ax= mos_df.plot.scatter(x='X',y='Y',c='g',label = 'Mosquito Source',ax=ax)
wat_df.plot.scatter(x='X',y='Y',c='m',figsize=(15, 15),label = 'Water Source',ax=ax)


# In[131]:

data_df.describe()


# In[132]:

Querycounty = data_df.groupby('YEAR')['COUNTRY'].value_counts()
county_df = Querycounty.unstack()
county_df.fillna(0,inplace=True)


# In[133]:

Querycounty = data_df.groupby('COUNTRY')
top10county = data_df.groupby('COUNTRY')['infection_source'].value_counts().sort_values(inplace=False)[::-1][1:10]
top10county.plot(kind='bar',title ="Top 10 effected Regions/Countries")


# In[134]:

sns.factorplot('infection_source_ID','infection_time',hue='VECTOR',data=data_df)


# In[135]:

Querycounty.aggregate(np.sum).plot()


# In[136]:

data_df['infection_source'].value_counts().plot(title ="Most influence source")


# In[137]:

data_df.hist( figsize=(15,15))
plt.show()


# In[138]:

names = data_df.columns


# In[139]:

import numpy 
import seaborn as sns
fig,ax = plt.subplots(figsize=(10,8))
correlations = data_df.corr()
sns.heatmap(correlations,ax=ax,vmax=1,square=True)
plt.title('Feature heatmap')


# In[140]:

data_df.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False, figsize=(15,15))
plt.show()


# In[141]:

from pandas.tools.plotting  import scatter_matrix
colors=['red','green']
plt.figure(figsize=(10, 10))
#scatter_matrix(data_df,figsize=[20,20],c=colors)
#plt.show()


# In[142]:

data_df.plot(kind='density', subplots=True, sharex=False,figsize=(10, 10))

plt.show()


# In[144]:

clusters = np.array(y_kmeans)
#X + clusters
clusters_df.loc[:,'clusters'] = clusters
clusters_df['COUNTRY'] = data_df_Y['COUNTRY']

Mosteffected = clusters_df[clusters_df['clusters'] == 0][['COUNTRY','X','Y','infection_time','infection_source_ID']]
writer = pd.ExcelWriter('Effected-Countries.xlsx', engine='xlsxwriter')
Mosteffected.to_excel(writer, sheet_name='High-risk Regions-countries')
Mosteffected = clusters_df[clusters_df['clusters'] == 1][['COUNTRY','X','Y','infection_time','infection_source_ID']]
Mosteffected.to_excel(writer, sheet_name='Risk-1 Regions-countries')
Mosteffected = clusters_df[clusters_df['clusters'] == 2][['COUNTRY','X','Y','infection_time','infection_source_ID']]
Mosteffected.to_excel(writer, sheet_name='Risk-2 Regions-countries')
Mosteffected = clusters_df[clusters_df['clusters'] == 3][['COUNTRY','X','Y','infection_time','infection_source_ID']]
Mosteffected.to_excel(writer, sheet_name='Risk-3 Regions-countries')

writer.save()



# In[ ]:



