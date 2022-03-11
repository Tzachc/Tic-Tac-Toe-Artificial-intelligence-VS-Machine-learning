#Final project Machine Learning.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.linear_model import LinearRegression
import matplotlib as mpl

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn import tree, metrics


data = pd.read_csv('tic-tac-toe-endgame.csv')
#print(data)
#display win or lose distrabution.
#sns.countplot(data['V10'])
#plt.show()

def onehot_encode(df, columns):
    df = df.copy()
    for column in columns:
        dummies = pd.get_dummies(df[column], prefix=column)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df


def preprocess_inputs(df):
    df = df.copy()

    # Encode label values as numbers
    df['V10'] = df['V10'].replace({'negative': 0, 'positive': 1})

    # One-hot encode board space columns
    df = onehot_encode(
        df,
        columns=['V' + str(i) for i in range(1, 10)]
    )

    # Split df into X and y
    y = df['V10'].copy()
    X = df.drop('V10', axis=1).copy()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,random_state=123)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_inputs(data)


###### start training #######
DecisionTree_2 = DecisionTreeClassifier(criterion='entropy')
DecisionTree = DecisionTreeClassifier()
DecisionTree.fit(X_train,y_train)
Decision_Tree = "{:.2f}%".format(DecisionTree.score(X_test,y_test) * 100)
DecisionTree_2.fit(X_train,y_train)
DecisionTree_2_v =  "{:.2f}%".format(DecisionTree_2.score(X_test,y_test) * 100)
names = ['', 'Gini', 'Entropy']
values = ['0', Decision_Tree,DecisionTree_2_v]
#plt.barh(names,values,color=['black','red'])
#plt.bar(names, values,color=['green','black', 'red'])
#plt.show()
print("Decision Tree " + "trained!")
print("Decision Tree accuracy: {:.2f}%" .format(DecisionTree.score(X_test,y_test) * 100))


Logistic_Regression  = LogisticRegression()
Logistic_Regression.fit(X_train,y_train)
y_pred = Logistic_Regression.predict(X_test)
#288 elements in X_test.

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
#confusion matrix to see the miscalifications.
print(cnf_matrix)

#print("Recall:",metrics.recall_score(y_test, y_pred,average="macro"))

print("Accuracy: ","{:.2f}%".format(metrics.accuracy_score(y_test, y_pred) *100))
#print("Precision:",metrics.precision_score(y_test, y_pred))

Linear_reg = LinearRegression()
Linear_reg.fit(X_train,y_train)
print("Linear Regression accuracy: {:.2f}%" .format(Linear_reg.score(X_test,y_test) * 100))

#PLOT THE MATRIX
'''
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
'''
print("Logistic Regression " + "trained!")
print("Logistic Regression accuracy: {:.2f}%" .format(Logistic_Regression.score(X_test,y_test) * 100))
Logistic_Regressions = "{:.2f}%".format(Logistic_Regression.score(X_test,y_test) * 100)


Adaboost = AdaBoostClassifier()
Adaboost.fit(X_train,y_train)

print("Adaboost " + "trained!")
print("Adaboost accuracy: {:.2f}%" .format(Adaboost.score(X_test,y_test) * 100))
AdaBoosts = "{:.2f}%".format(Adaboost.score(X_test,y_test) * 100)

#print(Adaboost.estimator_errors_)
#print(Adaboost.estimators_)
predictions = Adaboost.predict(X_test)
print(predictions)
print("Accuracy:",metrics.accuracy_score(y_test, predictions))

print(classification_report(y_test,predictions))
print (f'Train Accuracy - : {Adaboost.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {Adaboost.score(X_test,y_test):.3f}')
print("confusion matrix for adaboost classifier")
#displr = plot_confusion_matrix(Adaboost, X_test, y_test ,cmap=plt.cm.Blues , values_format='d')
#plt.show()

KNN = KNeighborsClassifier()
KNN.fit(X_train,y_train)
print("K-Nearest Neighbors " + "trained!")
print("K-Nearest Neighbors accuracy: {:.2f}%" .format(KNN.score(X_test,y_test) * 100))
K_Neighbors = "{:.2f}%" .format(KNN.score(X_test,y_test) * 100)


Support_Vector_Machine = SVC(kernel='linear')
Support_Vector_Machine.fit(X_train,y_train)
print("Support Vector Machine " + "trained!")
print("Support Vector Machine accuracy: {:.2f}%" .format(Support_Vector_Machine.score(X_test,y_test) * 100))
SVM = "{:.2f}%" .format(Support_Vector_Machine.score(X_test,y_test) * 100)


dict = {'': '0', 'Decision Tree': Decision_Tree, 'Logistic Regression':Logistic_Regressions,'Adaboost':AdaBoosts,'KNN':K_Neighbors,'SVM':SVM}
#names = ['Decision Tree', 'Logistic Regression', 'Adaboost', 'KNN','SVM']
#values = [Decision_Tree, Logistic_Regressions, AdaBoosts, K_Neighbors,SVM]
#values.sort()

sorted_dict = {}
sorted_keys = sorted(dict, key=dict.get)  # [1, 3, 2]

for w in sorted_keys:
    sorted_dict[w] = dict[w]
#plt.figure(figsize=(20, 3),dpi= 80)
#plt.subplot(131)
'''
PLOT THE OUTPUTS RESULTS.
'''
#plt.bar(sorted_dict.keys(), sorted_dict.values(),color=['black', 'red', 'green', 'blue', 'cyan'])
#plt.show()

#print(wrong_examples)
#print(wrong_examples)
def print_wrong_examples(df):
    wrong_examples = X_test.loc[(df.predict(X_test) != y_test), :]
    wrong_examples = data.loc[wrong_examples.index, :].drop('V10', axis=1)
    k = 0
    for i in wrong_examples.index:
        print("\nExample " + str(i))
        print(wrong_examples.loc[i, 'V1'] + " " + wrong_examples.loc[i, 'V2'] + " " + wrong_examples.loc[i, 'V3'])
        print(wrong_examples.loc[i, 'V4'] + " " + wrong_examples.loc[i, 'V5'] + " " + wrong_examples.loc[i, 'V6'])
        print(wrong_examples.loc[i, 'V7'] + " " + wrong_examples.loc[i, 'V8'] + " " + wrong_examples.loc[i, 'V9'])
        k += 1
    print("\n" + "Count: " + str(k))
#print_wrong_examples(DecisionTree)
#print_wrong_examples(DecisionTree)
#print("############## LOGISTIC ############")
print_wrong_examples(Logistic_Regression)
print("#########################################" + "\n")
print_wrong_examples(Support_Vector_Machine)
#print_wrong_examples(DecisionTree_2)
#wrong_examples = X_test.loc[(Logistic_Regression.predict(X_test) != y_test), :]
#wrong_examples = data.loc[wrong_examples.index, :]
#print(wrong_examples)
#print_wrong_examples(Support_Vector_Machine)
#print("########## LOGISTIC ########")
#print_wrong_examples(Logistic_Regression)
def visualizing_tree(df):
    plt.figure(figsize=(30,30))
    tree.plot_tree(df, filled=True)
    plt.show()


#depth = DecisionTree_2.get_depth()
#print(depth)
#visualizing_tree(DecisionTree_2)
