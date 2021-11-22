from sklearn.model_selection import train_test_split

from functions import RadiusFunction, LearningRateFunction, standarize
from kohonen_network import KohonenNetwork
import pandas as pd
from metrics import ConfusionMatrix

PREDICTION = 1
LABEL = 0


def get_confusion_matrix(results):
    confusion_matrix = ConfusionMatrix(['0', '1'])
    for result in results:
        confusion_matrix.add_entry(result[LABEL], result[PREDICTION])
    confusion_matrix.summarize()
    confusion_matrix.print_confusion_matrix()



df = pd.read_csv('acath.csv').dropna()
real_classifications = df['sigdz'].head(20).to_numpy()
df = df.drop(['sigdz'], axis=1)
df = df.drop(['sex'], axis=1)
df = df.drop(['tvdlm'], axis=1)

data = standarize(df.head(20))

network_size = int(len(data) / 5)  # 3 x 3
max_radius = network_size
max_learning_rate = 0.1
iterations = 5 * network_size
radius = RadiusFunction(network_size, iterations)
learning_rate = LearningRateFunction(max_learning_rate, iterations)
input_size = 3

x_train, x_test, y_train, y_test = train_test_split(data, real_classifications, test_size=0.3, random_state=0)

network = KohonenNetwork(network_size, input_size, "rectangular")
network.initialize(x_train, y_train)
network.train(iterations, learning_rate, radius)
network.plot_clusters()
network.build_U_matrix()
network.plot_network_hit_map()
# network.plot_network_features(['age', 'cad.dur', 'choleste'])

results = network.predict(x_test, y_test)
get_confusion_matrix(results)
