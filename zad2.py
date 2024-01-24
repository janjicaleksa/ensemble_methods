import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

names = ['x'+str(i) for i in range(13)]
names.append('y')
df = pd.read_csv('data_1.csv', header=None, names=names)

X_train, X_test, y_train, y_test = train_test_split(df[['x0', 'x3']], df['y'], test_size=0.25, random_state=22, stratify=df['y'])
max_depth_list = [i for i in range(1, 11)]
acc_train = []
acc_val = []
for max_depth in max_depth_list:
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf = clf.fit(np.array(X_train), y_train)
    acc_train.append(clf.score(np.array(X_train), y_train))
    acc_val.append(clf.score(np.array(X_test), y_test))

plt.figure(figsize=(8, 4))
plt.plot(max_depth_list, acc_train, c='b', marker='o', label='Accuracy-training')
plt.plot(max_depth_list, acc_val, c='r', marker='o', label='Accuracy-test')
plt.xlabel('max depth')
plt.ylabel('accuracy')
plt.legend()
plt.show()
def decision_tree(X_train, y_train, max_depth, figsize):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf = clf.fit(np.array(X_train), y_train)

    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    _ = tree.plot_tree(clf, feature_names=X_train.columns, class_names=['0', '1'], filled=True, fontsize=14)

    plt.figure(figsize=(8, 4))
    plt.scatter(X_train['x0'], X_train['x3'], c=np.array(y_train), cmap=plt.cm.jet)
    xmin, xmax, ymin, ymax = plt.axis()
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, num=100, endpoint=True), np.linspace(ymin, ymax, num=100, endpoint=True))
    y_pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    y_pred = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, alpha=0.2, cmap='bwr')
    plt.xlabel('x0')
    plt.ylabel('x3')
    plt.title('Decision boundary and training data samples')
    plt.show()

decision_tree(X_train, y_train, 1, [4, 4])
decision_tree(X_train, y_train, 2, [6, 6])
decision_tree(X_train, y_train, 6, [50, 25])