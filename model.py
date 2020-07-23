#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 08:40:02 2019

@author: arv
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv("data.csv")
X = dataset.iloc[: , [0,1,2]].values
y = dataset.iloc[: , -1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.17,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#fitting KNN regression

from sklearn.neighbors import KNeighborsClassifier
classifier =  KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)

#y_pred1 = classifier.predict(sc_X.transform([[19,1,1]]))
#predicting the X_test result
#xyz = [[19,1,1]]
#xyz = sc_X.transform(xyz)
#print(xyz)

#y_pred = classifier.predict(sc_X.transform([[19,1,1]]))
#y_pred = classifier.predict(xyz)

pickle.dump(classifier, open('model_sdl.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model_sdl.pkl','rb'))
#sc_X.fit()
print(model.predict(sc_X.transform([[19,1,1]])))

#making confusing matrix

"""from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)"""

"""#visualising the training set
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2, X3 , X4 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 2].min() - 1, stop = X_set[:, 2].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 3].min() - 1, stop = X_set[:, 3].max() + 1, step = 0.01))
plt.contourf(X1, X2, X3, X4, classifier.predict(np.array([X1.ravel(), X2.ravel(), X3.ravel(),X4.ravel())]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue','green'))(i), label = j)
plt.title('KNN Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#visualising test set result

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()"""