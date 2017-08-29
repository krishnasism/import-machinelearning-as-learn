from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

def dist(x,y):
    return np.sqrt((np.sum(x-y)**2)) #find the distance between two points in a 2 axis graph


digits = datasets.load_digits() #load images of digits datasets 
print(digits.images[0]) 
print(digits.target[0])

index = 1

#create a subset of the database for the training set
#10 images in dataset 0-9
X_train = digits.data[0:1000]
Y_train = digits.target[0:1000]

#create a test image
X_test = digits.data[index] #get a random data for input

#show the test image                       
#plt.figure()
#plt.imshow(digits.images[index],cmap=plt.cm.gray_r,interpolation='nearest')
#plt.show()

errors = 0
num = len(X_train)
distance = np.zeros(num)

for j in range(1697, 1797):
    X_test = digits.data[j]
    for i in range(num):
        distance[i]=dist(X_train[i],X_test)
    min_index = np.argmin(distance)
    if Y_train[min_index] != digits.target[j]:
        errors +=1
        
print("ERRORS")
print(errors)

min_index = np.argmin(distance)
print("NUMBER : ")
print(Y_train[min_index])




