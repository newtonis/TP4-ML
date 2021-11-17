import numpy as np
import pandas as pd

from K_means import KMeans

df = pd.read_excel('./acath.xls').dropna()
# df = df.replace(r'^\s+$', np.nan, regex=True).dropna(how='all')
real_classifications = df['sigdz'].head(10).to_numpy()
df = df.drop(['sigdz'], axis=1)
data = df.head(10).to_numpy()

k = 2
model = KMeans(k)
model.fit(data)
model.print_result(real_classifications)