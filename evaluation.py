import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def evaluate(data):
    model_target = data.columns[-1]     #Selecting the last label as target
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=[model_target], axis=1),
    data[model_target],
    test_size=0.3,
    random_state=None)

    #evalute using decision tree
    ModelDT = DecisionTreeClassifier(
                            criterion='gini',
                            splitter='best'
                            )
    ModelDT.fit(X_train, y_train)
    predictionsDT = ModelDT.predict(X_test)
    accuracyDT = accuracy_score(y_test, predictionsDT)


    #evaluate using svm
    ModelSVM = SVC(random_state=1)
    ModelSVM.fit(X_train, y_train)
    predictionsSVM = ModelSVM.predict(X_test)
    accuracySVM = accuracy_score(y_test, predictionsSVM)


    #evaluate using knn
    ModelKNN = KNeighborsClassifier()
    ModelKNN.fit(X_train, y_train)
    predictionsKNN = ModelKNN.predict(X_test)
    accuracyKNN = accuracy_score(y_test, predictionsKNN)

    score = (accuracyDT + accuracySVM + accuracyKNN)/3

    return score

 
