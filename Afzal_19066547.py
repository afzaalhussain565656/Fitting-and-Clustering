#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score





dataset = pd.read_csv('wine-clustering.csv')
dataset.dropna(inplace = True)
print(dataset)
print(dataset.shape)



dataset.dropna(inplace = True)
scaler = StandardScaler()

dataset[['Alcohol_T','Malic_Acid_T','Ash_T','Ash_Alcanity_T','Magnesium_T','Total_Phenols_T','Flavanoids_T','Nonflavanoid_Phenols_T','Proanthocyanins_T','Color_Intensity_T','Hue_T','OD280_T','Proline']] = scaler.fit_transform(dataset[['Alcohol','Malic_Acid','Ash','Ash_Alcanity','Magnesium','Total_Phenols','Flavanoids','Nonflavanoid_Phenols','Proanthocyanins','Color_Intensity','Hue','OD280','Proline']])


# In[2]:


def optimize_k_mean(data, max_k):
    means=[]
    inertias=[]
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        
        means.append(k)
        inertias.append(kmeans.inertia_)
        
    fig = plt.subplots(figsize=(10,5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()


# In[3]:


def optimise():
    optimize_k_mean(dataset[['Alcohol_T','Malic_Acid_T']], 10)
    Kmeans = KMeans(n_clusters=3)
    Kmeans.fit(dataset[['Alcohol_T','Malic_Acid_T']])
    dataset['kmeans_3'] = Kmeans.labels_


# In[4]:


optimise()


# In[5]:


#Line fitting

def line_fitting():
    x = dataset['Alcohol_T']
    y = dataset['Malic_Acid_T']

    #Here find the line which best fits
    x_1, y_1 = np.polyfit(x, y, 1)

    #poitns scattering
    plt.scatter(x, y, color='green')

    #Now adding best line which fits in the graph
    m = (x_1*x+y_1)
    plt.plot(x, m, color='red', linestyle='--', linewidth=3)


# In[6]:


line_fitting()


# In[7]:


#Scatter Plot

def scatter_plot():
    plt.scatter(x=dataset['Alcohol_T'], y=dataset['Malic_Acid_T'], c=dataset['kmeans_3'])
    plt.show()
    


# In[8]:


scatter_plot()


# In[9]:


def histogram():
    plt.title("Histogram of Clustering and Fitting", fontsize=12)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.hist(dataset['Alcohol_T'], bins=10, edgecolor = 'black')
    plt.show()
    


# In[10]:


histogram()


# In[11]:


import seaborn as sn

def heatmap():
    hm = sn.heatmap(dataset) 
    # displaying the plotted heatmap 
    plt.show()


# In[12]:


heatmap()


# In[ ]:




