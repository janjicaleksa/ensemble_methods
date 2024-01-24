import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

df = pd.read_csv('data_2.csv', header=None, names=['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'y'])

#Random Forest
num_estimators_list = [10, 20, 50, 100]
max_depth_list = [2, 4, 6, 8, 10, 12, 14]
max_features_list = [1, 2, 3, 4, 5]
def crossvalidation_rf(df, n_estimators, max_depth, max_features, k):
    X = df.drop('y', axis=1)
    y = df['y']
    m = len(y)
    f1_scores = np.zeros(k)
    for i in range(k):
        X_test = X[i*m//k:(i+1)*m//k]
        y_test = y[i*m//k:(i+1)*m//k]
        X_train = np.concatenate((X[0:i*m//k], X[(i+1)*m//k:]), axis=0)
        y_train = np.concatenate((y[0:i*m//k], y[(i+1)*m//k:]), axis=0)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(np.array(X_test))
        f1_scores[i] = f1_score(y_test, y_pred)
    return np.mean(f1_scores)

f1_mean = np.zeros([len(num_estimators_list), len(max_depth_list), len(max_features_list)])
for i in range(len(num_estimators_list)):
    for j in range(len(max_depth_list)):
        for k in range(len(max_features_list)):
            f1_mean[i, j, k] = crossvalidation_rf(df, num_estimators_list[i], max_depth_list[j], max_features_list[k], 7)

ind_max = np.unravel_index(np.argmax(f1_mean, axis=None), f1_mean.shape)
print(f"Optimal parameters: ensemble size: {num_estimators_list[ind_max[0]]},"f" max depth: {max_depth_list[ind_max[1]]}, max features: {max_features_list[ind_max[2]]} ")

plt.figure(figsize=(8, 4))
plt.plot(num_estimators_list, f1_mean[:, ind_max[1], ind_max[2]], marker='o')
plt.xlabel('ensemble size')
plt.ylabel('F1-score')
plt.title('Dependency of the F1-score on the ensemble size')
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(max_depth_list, f1_mean[ind_max[0], :, ind_max[2]], marker='o')
plt.xlabel('max depth')
plt.ylabel('F1-score')
plt.title('Dependency of the F1-score on the max depth')
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(max_features_list, f1_mean[ind_max[0], ind_max[1], :], marker='o')
plt.xlabel('max features')
plt.ylabel('F1-score')
plt.title('Dependency of the F1-score on the max features')
plt.show()

#Gradient Boost
num_estimators_list = [10, 20, 50, 100, 200]
max_depth_list = [2, 4, 6, 8, 10, 12, 14]
learning_rate_list = [0.01, 0.05, 0.1, 0.5]
def crossvalidation_gb(df, n_estimators, max_depth, learning_rate, k):
    X = df.drop('y', axis=1)
    y = df['y']
    m = len(y)
    f1_scores = np.zeros(k)
    for i in range(k):
        X_test = X[i*m//k:(i+1)*m//k]
        y_test = y[i*m//k:(i+1)*m//k]
        X_train = np.concatenate((X[0:i*m//k], X[(i+1)*m//k:]), axis=0)
        y_train = np.concatenate((y[0:i*m//k], y[(i+1)*m//k:]), axis=0)
        clf = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(np.array(X_test))
        f1_scores[i] = f1_score(y_test, y_pred)
    return np.mean(f1_scores)

f1_mean = np.zeros([len(num_estimators_list), len(max_depth_list), len(learning_rate_list)])
for i in range(len(num_estimators_list)):
    for j in range(len(max_depth_list)):
        for k in range(len(learning_rate_list)):
            f1_mean[i, j, k] = crossvalidation_gb(df, num_estimators_list[i], max_depth_list[j], learning_rate_list[k], 7)

ind_max = np.unravel_index(np.argmax(f1_mean, axis=None), f1_mean.shape)
print(f"Optimal parameters: ensemble size: {num_estimators_list[ind_max[0]]},"f" max depth: {max_depth_list[ind_max[1]]}, learning rate: {learning_rate_list[ind_max[2]]} ")

plt.figure(figsize=(8, 4))
plt.plot(num_estimators_list, f1_mean[:, ind_max[1], ind_max[2]], marker='o')
plt.xlabel('ensemble size')
plt.ylabel('F1-score')
plt.title('Dependency of the F1-score on the ensemble size')
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(max_depth_list, f1_mean[ind_max[0], :, ind_max[2]], marker='o')
plt.xlabel('max depth')
plt.ylabel('F1-score')
plt.title('Dependency of the F1-score on the max depth')
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(learning_rate_list, f1_mean[ind_max[0], ind_max[1], :], marker='o')
plt.xlabel('learning rate')
plt.ylabel('F1-score')
plt.title('Dependency of the F1-score on the learning rate')
plt.show()