import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn import svm

def train_and_predict(model, features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("accuracy score: %.2f" % accuracy_score(y_test, y_pred))
    returned = confusion_matrix(y_test, y_pred).ravel()
    print("confusion matrix")
    print(returned)


df = pd.read_csv("dataset_final.csv")

features = df.drop(['State',"Championships"], axis=1)
target = df[['Championships']]

# Decision Tree
naive_model = DecisionTreeClassifier(random_state=42)
param_model = DecisionTreeClassifier(max_depth=5, min_samples_split=3, min_samples_leaf=2, random_state=42)

train_and_predict(naive_model, features, target)
train_and_predict(param_model, features, target)

export_graphviz(naive_model, out_file = "tree.dot")
export_graphviz(param_model, out_file = "tree2.dot")

# SVM
# using train value also as test value
a_scores = 0
for i in range(100):
    X = df.drop(['State',"Championships"], axis=1)
    Y = df[['Championships']]
    model = SVC()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    model.fit(X_train, Y_train.values.ravel())
    predicted = model.predict(X_train)
    expected = Y_train
    score = metrics.accuracy_score(expected, predicted)
    a_scores += score

a_score = a_scores / 100
print(a_score)

a_scores = 0
for i in range(100):
    X = df.drop(['State',"Championships"], axis=1)
    Y = df[['Championships']]
    model = SVC()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    model.fit(X_train, Y_train.values.ravel())
    predicted = model.predict(X_test)
    expected = Y_test 
    score = metrics.accuracy_score(expected, predicted)
    a_scores += score

a_score = a_scores / 100
print(a_score)

a_scores = {}
for i in np.linspace(0.1, 1, 9):
    X = df.drop(['State',"Championships"], axis=1)
    Y = df[['Championships']]
    model = SVC(C=i)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    model.fit(X_train, Y_train.values.ravel())
    predicted = model.predict(X_test)
    #predicted = model.predict(X_train)
    expected = Y_test 
    #expected = Y_train
    score = metrics.accuracy_score(expected, predicted)
    a_scores[i] = score

plt.plot(a_scores.keys(), a_scores.values())
plt.xlabel('Regularization Parameter')
plt.ylabel('Accuracy Scores')
plt.title('Regularization Tuning')
plt.savefig("regularization_tuning.png")

