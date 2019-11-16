# Classifying with k-Nearest Neighbors

## Introduction

k-Nearest Neighbors (kNN) is  an effective way to classify with distance measurements.

* Pros: High accuracy,insensitive to outliers,no assumptions about data

* Cons: Computationally expensive,requires a lot of memory

* Works with: Numeric values,nominal values

  

## principle

* prepare data

  get some input data and output structured numeric values.

* distance calculation

  find the characteristic value and use the Euclidian distance to calculate the distance between two vectors.

* sort the distance results in increasing order 
* take k items with lowest distances to inX
* find the majority class among these items
* return the majority class as our prediction for the class of inX

## application

``` python
# prepare
from sklearn import datasets 
from sklearn.neighbors import KNeighborsClassifier
import numpy as np `
np.random.seed(0)
```

``` python
# data processing(using iris)
iris=datasets.load_iris() 
iris_x=iris.data #attributes
iris_y=iris.target  
indices = np.random.permutation(len(iris_x)) 
```

``` python
# Training and test sets
iris_x_train = iris_x[indices[:-10]]
iris_y_train = iris_y[indices[:-10]] 
iris_x_test  = iris_x[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]] 
```

``` python
# model training
knn = KNeighborsClassifier() 
knn.fit(iris_x_train, iris_y_train)  
```

``` python
# model prediction
iris_y_predict = knn.predict(iris_x_test) 
probility=knn.predict_proba(iris_x_test)  
neighborpoint=knn.kneighbors(iris_x_test[-1:],5,False)
score=knn.score(iris_x_test,iris_y_test,sample_weight=None)

```

``` python
# print results
print('iris_y_predict = ')  
print(iris_y_predict)  
print('iris_y_test = ')
print(iris_y_test)    
print ('Accuracy:',score  )
print ('neighborpoint of last test sample:',neighborpoint)
print ('probility:',probility)
```



reference:https://blog.csdn.net/andy_shenzl/article/details/82800726





