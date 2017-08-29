import numpy as np
import matplotlib.pyplot as plt

X_train=np.array([[1,1],[2,2.5],[3,1.2],[5.5,6.3],[6,9],[7,6]])
Y_train=['red','red','red','blue','blue','blue']
X_test=np.array([3,4])
plt.figure()
plt.scatter(X_train[:,0],X_train[:,1],s=170,color=Y_train[:])
plt.scatter(X_test[0],X_test[1],s=170,color='green')
plt.show()

num = len(X_train) #Number of points in the X_train array
distance = np.zeros(num) #empty array w/ 0s

def dist(x,y):
    return np.sqrt((np.sum(x-y)**2)) #find the distance between two points in a 2 axis graph

for i in range(num):
    distance[i]=dist(X_train[i],X_test) #finds the distance delta between the test point and a point in X_train array
print(distance) 

min_index = np.argmin(distance)
print(Y_train[min_index])
