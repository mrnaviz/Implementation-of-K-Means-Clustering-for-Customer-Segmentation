# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries,read the data set.find the number of null data.

2. Find the number of null data.

3. Import sklearn library.

4. Find y predict and plot the graph.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: NAVEEN KUMAR B
RegisterNumber:  212222230091

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers (1).csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[]

for i in range (1,11):
    kmeans=KMeans(n_clusters = i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow matter")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segmets")
*/
```

## Output:
data.head():

![image](https://github.com/mrnaviz/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/123350791/059e044b-4948-418e-973c-251cb28b42e9)

data.info():

![image](https://github.com/mrnaviz/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/123350791/b8d2a9d6-fe7a-4e46-a367-b73840d844f7)

data.isnull().sum():

![image](https://github.com/mrnaviz/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/123350791/b0a02a5d-0964-41ac-ba5c-451814bf62c8)

Elbow method:

![image](https://github.com/mrnaviz/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/123350791/1fa17e01-a680-4725-b945-4e26d1f5fd74)

K-Means:

![image](https://github.com/mrnaviz/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/123350791/704b334f-6919-4686-8aed-314376d1e4e7)

Array value of Y:

![image](https://github.com/mrnaviz/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/123350791/9340f76a-53e9-4876-8be8-70ace57183a5)

Customer Segmentation:

![image](https://github.com/mrnaviz/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/123350791/cdc9d504-0246-4f74-b12c-c31e2e6dbf3b)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
