import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

classifiers = [Perceptron(),DecisionTreeClassifier(),SVC(),KNeighborsClassifier(),QuadraticDiscriminantAnalysis(),GaussianNB(),
               RandomForestClassifier(),AdaBoostClassifier()]

names = ["Perceptron","Decision Tree","Linear SVM","Nearest Neighbors","QDA","Gaussian Process","Random Forest","AdaBoost"]
   
#[height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],[190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female','female', 'male', 'male']
acc = np.zeros(len(classifiers))

for i in range(len(classifiers)):
    exec('clf'+str(i)+' = classifiers['+str(i)+'].fit(X, y)')
    exec('prediction = clf'+str(i)+'.predict(X)')
    accuracy = accuracy_score(y, prediction) * 100
    acc[i] = accuracy

# The best classifier
index = np.argmax(acc)
print prediction[index]
print 'Best classifier is ',str(classifiers[index]).rsplit('(',1)[0]
