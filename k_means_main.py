import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from K_means import KMeans
from metrics import ConfusionMatrix

PREDICTION = 2
LABEL = 1


def get_confusion_matrix(results):
    confusion_matrix = ConfusionMatrix(['0', '1'])
    for result in results:
        confusion_matrix.add_entry(result[LABEL], result[PREDICTION])
    confusion_matrix.summarize()
    confusion_matrix.print_confusion_matrix()


df = pd.read_csv('acath.csv').dropna()
real_classifications = df['sigdz'].to_numpy()
df = df.drop(['sigdz'], axis=1)
df = df.drop(['sex'], axis=1)
df = df.drop(['tvdlm'], axis=1)

data = df.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(data, real_classifications, test_size=0.3, random_state=0)

k = 2
model = KMeans(k)
model.fit(x_train, y_train, centroid_assignment='distance')
train_results = model.sarasa()
model.plot_result(0, 0, ['age', 'cad.dur', 'choleste'])
model.plot_result(-60, 0, ['age', 'cad.dur', 'choleste'])
model.plot_result(60, 0, ['age', 'cad.dur', 'choleste'])

#get_confusion_matrix(train_results)

#predictions = model.predict(x_test)
#test_results = np.stack((np.arange(len(x_test)), predictions, y_test), axis=1)
#get_confusion_matrix(test_results)

