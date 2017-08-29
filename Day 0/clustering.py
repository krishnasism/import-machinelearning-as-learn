import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.array([[1,1],[2,2.5],[3,1.2],[5.5,6.3],[6,9],[7,6],[8,8]])
plt.figure()
plt.scatter(X[:,0],X[:,1],s=170,color="black")
plt.show()
k=3
kmeans = KMeans(n_clusters = k) #run kmeans algorithm
kmeans.fit(X)  #use kmeans algorithm to fit X to graph
centroids =kmeans.cluster_centers_  #find the centroids between the groups
labels = kmeans.labels_

colors = ['r.','g.','y.']
plt.figure()
for i in range(len(X)):
    plt.plot(X[i,0], X[i,1], colors[labels[i]], markersize = 30)
plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=300,linewidths=5)
plt.show()

