import numpy as np 
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

names = ['x'+str(i) for i in range(13)]
names.append('y')
df = pd.read_csv('data_1.csv', header=None, names=names)

# Feature selection - correlation
corr = df.corr()
sb.set(rc={'figure.figsize': (15, 8)})
sb.heatmap(corr, annot=True)

corr_array = np.abs(np.array(corr['y'].drop('y')))
names = ['x'+str(i) for i in np.argsort(corr_array)[::-1]]
plt.figure(figsize=(12, 4))
plt.bar(names, np.flip(np.sort(corr_array)), color='blue', width=0.5)
plt.title('Absolute values of correlation with output for different features')
plt.show()

# Wrapper feature selection
def crossvalidation(X_train, y_train, C_list, k):
    X = np.array(X_train)
    y = np.array(y_train)
    m = len(y)
    acc = []
    for C in C_list:
        sum_acc = 0
        for i in range(k):
            X_test = X[i*m//k:(i+1)*m//k]
            y_test = y[i*m//k:(i+1)*m//k]
            X_all = np.concatenate((X[0:i*m//k], X[(i+1)*m//k:]), axis=0)
            y_all = np.concatenate((y[0:i*m//k], y[(i+1)*m//k:]), axis=0)
            clf = LogisticRegression(random_state=0, penalty='l2', C=C).fit(X_all, y_all)
            y_pred = clf.predict(X_test)
            sum_acc += sum(y_pred==y_test)/len(y_test)
        acc.append(sum_acc/k)

    best_index = np.argmax(np.array(acc))
    C_best = C_list[best_index]
    return C_best
def wrapper_method(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['y'], axis=1), df['y'], test_size=0.3, random_state=42, stratify=df['y'])

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    C_list = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    X = []
    acc = []
    for i in range(X_train.shape[1]):
        acc_curr = np.zeros(X_train.shape[1])
        for j in range(X_train.shape[1]):
            if j not in X:
                if i == 0:
                    X_train_curr = X_train[:, j].reshape(-1, 1)
                    X_test_curr = X_test[:, j].reshape(-1, 1)
                else:
                    X_train_curr = X_train[:, X+[j]]
                    X_test_curr = X_test[:, X+[j]]

                C_best = crossvalidation(X_train_curr, y_train, C_list, 3)
                clf = LogisticRegression(random_state=0, penalty='l2', C=C_best).fit(X_train_curr, y_train)
                y_pred = clf.predict(X_test_curr)
                acc_curr[j] = sum(y_pred==y_test)/len(y_test)

        ind_max = np.argmax(acc_curr)
        X.append(ind_max)
        acc.append(acc_curr[ind_max])
    return acc, X

acc, X = wrapper_method(df)
names = ['x'+str(i) for i in X]
plt.figure(figsize=(12, 4))
plt.plot(names, acc, marker='o')
plt.title('Wrapper feature selection - accuracy')
plt.show()