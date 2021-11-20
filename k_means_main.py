import numpy as np
import pandas as pd

from K_means import KMeans
from metrics import ConfusionMatrix

PREDICTION = 2
LABEL = 1

df = pd.read_csv('acath.csv').dropna()
# df = df.replace(r'^\s+$', np.nan, regex=True).dropna(how='all')
real_classifications = df['sigdz'].head(10).to_numpy()
df = df.drop(['sigdz'], axis=1)

print(df)

data = df.head(10).to_numpy()

k = 2
model = KMeans(k)
model.fit(data)
# model.print_result(real_classifications)
results = model.sarasa()
confusion_matrix = ConfusionMatrix(['0', '1'])
for result in results:
    confusion_matrix.add_entry(result[LABEL], result[PREDICTION])
confusion_matrix.summarize()
confusion_matrix.print_confusion_matrix()
