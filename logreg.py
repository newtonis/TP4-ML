import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import math
import matplotlib.pyplot as plt

logreg = LogisticRegression(solver='lbfgs', random_state=0, verbose=1)

content = pd.read_csv("acath.csv")
content = content.to_dict("records")
print(content[0].keys())

X = np.zeros(shape=(len(content), 3))
Y = np.zeros(shape=(len(content)))

index = 0

for item in content:
    choeste = 0

    if not math.isnan(item["age"]) and not math.isnan(item["choleste"]) and not math.isnan(item["cad.dur"]):
        if not(item["age"] == 0 and item["choleste"] == 0 and item["cad.dur"] == 0):
            X[index, 0] = item["age"]
            X[index, 1] = item["choleste"]
            X[index, 2] = item["cad.dur"]

            Y[index] = item["sigdz"]
            print(item["age"], item["choleste"], item["cad.dur"], "=>", item["sigdz"])
            index += 1

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


ones = 0
zeros = 0

# for i in range(len(y_train)):
#     ones += int(y_train[i]) == 1
#     zeros += int(y_train[i]) == 0
#
# print("ones: %d" % ones)
# print("zeros: %d" % zeros)


logreg = logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_train)
y_prob = logreg.predict_proba(X_train)

prediccion_test = logreg.predict(np.array([[49, 220.0, 41]]))
print("test-prediction = ", prediccion_test)

for i in range(len(y_pred)):
    ones += int(y_pred[i]) == 1
    zeros += int(y_pred[i]) == 0
    print(y_prob[i])


print("ones: %d" % ones)
print("zeros: %d" % zeros)

print("params trained = ", logreg.coef_)

score = logreg.score(X_test, y_test)

print("model score = ", score)

confusion_matrix = [[0, 0], [0, 0]]


for item in range(y_test.shape[0]):
    confusion_matrix[int(y_pred[item])][int(y_test[item])] += 1


sns.heatmap(confusion_matrix, annot=True, fmt='g', xticklabels=[0, 1], yticklabels=[0, 1])

plt.title("Matriz de confusion")
plt.show()
