import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


def get_furthest_point(element_set, centroids):
    point_distances = {}
    for i in range(len(element_set)):
        point_distances[i] = 0  # inicializo acumulador de distancias al cuadrado en 0
        for j in range(len(centroids)):
            d = np.linalg.norm(element_set[i] - centroids[j]) ** 2
            point_distances[i] += d  # sumo la distancia de ese centroide al acumulador
    # me quedo con el indice del elemento con mayor suma distancia al cuadrado a todos
    furthest_index = max(point_distances, key=point_distances.get)
    return element_set[furthest_index]


class KMeansElement:
    def __init__(self, class_value, value, id, label):
        self.class_value = class_value
        self.value = value
        self.id = id
        self.label = label

    def get_id(self):
        return self.id

    def get_label(self):
        return self.label

    def get_class(self):
        return self.class_value

    def change_class(self, _class):
        self.class_value = _class

    def get_element(self):
        return self.value


class KMeansClass:
    def __init__(self, initial_centroid, class_value):
        self.class_value = class_value
        self.elements = []
        self.centroid = initial_centroid
        self.summary = None
        self.most_representative_label = None

    def set_centroid(self, centroid):
        self.centroid = centroid

    def get_elements(self):
        return self.elements

    def get_most_representative_label(self):
        if self.most_representative_label is None:
            self.calculate_most_representative_label()
        return self.most_representative_label

    def calculate_most_representative_label(self):
        if self.summary is None:
            self.summarize()
        self.most_representative_label = max(self.summary, key=self.summary.get)
        return self.most_representative_label

    def add_element(self, element):
        self.elements.append(element)

    def remove_element(self, element):
        for x in range(len(self.elements)-1, -1, -1):
            if (self.elements[x].get_element() == element.get_element()).all():
                del self.elements[x]
                break

    def get_centroid(self):
        return self.centroid

    def get_w(self):
        w = 0
        for i in range(self.get_elements_amount()):
            for j in range(i+1, self.get_elements_amount()):
                w = (self.elements[i].get_element() - self.elements[j].get_element()) ** 2
        if self.get_elements_amount() != 0:
            w /= self.get_elements_amount()
        else:
            w = 0
        return w

    def get_class_value(self):
        return self.class_value

    def get_distance_to_class(self, element):
        return np.linalg.norm(self.get_centroid() - element.get_element())

    def calculate_centroid(self):
        if len(self.elements) == 0:
            return self.centroid
        new_centroid = 0
        for element in self.elements:
            new_centroid += element.get_element()
        new_centroid /= len(self.elements)
        self.centroid = new_centroid
        return self.get_centroid()

    def get_elements_amount(self):
        return len(self.elements)

    def summarize(self):
        if self.summary is not None:
            return self.summary
        summary = {}
        for element in self.elements:
            if element.get_label() in summary:
                summary[element.get_label()] += 1
            else:
                summary[element.get_label()] = 1
        self.summary = summary
        return summary


class KMeans:
    def __init__(self, k):
        self.classes_amount = k
        self.classes = None
        self.elements = []
        self.stationary = False
        self.labels = None

    def get_accum_w(self):
        accum = 0
        for _class in self.classes:
            accum += _class.get_w()
        return accum

    def get_clusters_sumary(self):
        summary = []
        for _class in self.classes:
            summary.append(_class.summarize())
        return summary

    def sarasa(self):
        results = []
        for element in self.elements:
            result_for_element = [element.get_id(), element.get_label(),
                                  self.classes[element.get_class()].get_most_representative_label()]
            results.append(result_for_element)
        return results

    def max_distance_assignment(self, element_set, labels):
        centroids = []
        first_centroid = element_set[0]
        centroids.append(first_centroid)
        self.classes[0].set_centroid(first_centroid)
        for i in range(1, len(self.classes)):
            next_centroid = get_furthest_point(element_set, centroids)
            self.classes[i].set_centroid(next_centroid)
            centroids.append(next_centroid)
        self.point_assignment(element_set, labels)

    def initialize_centroids(self, centroid_assignment, element_set, labels):
        if centroid_assignment == 'random':
            self.random_assignment(element_set, labels)
        elif centroid_assignment == 'distance':
            self.max_distance_assignment(element_set, labels)

    def random_assignment(self, element_set, labels):
        for _class in self.classes:
            _class.set_centroid(element_set[random.randint(0, len(element_set)-1)])
        self.point_assignment(element_set, labels)

    def point_assignment(self, element_set, labels):
        index = 0
        for element in element_set:
            class_value = random.randint(0, self.classes_amount-1)
            self.classes[class_value].add_element(KMeansElement(class_value, element, index, labels[index]))
            self.elements.append(KMeansElement(class_value, element, index, labels[index]))
            index += 1

    def calculate_centroids(self):
        for _class in self.classes:
            _class.calculate_centroid()

    def get_closest_class(self, k_means_element):
        distances = {}
        for _class in self.classes:
            d = _class.get_distance_to_class(k_means_element)
            distances[_class.get_class_value()] = d
        closest_class = min(distances, key=distances.get)
        return closest_class

    def update_state(self):
        has_changes = False
        for k_means_element in self.elements:
            closest_class = self.get_closest_class(k_means_element)
            if closest_class != k_means_element.get_class():
                has_changes = True
                self.classes[k_means_element.get_class()].remove_element(k_means_element)
                self.classes[closest_class].add_element(k_means_element)
                k_means_element.change_class(closest_class)
        self.calculate_centroids()
        self.stationary = not has_changes

    def fit(self, element_set, labels=None, centroid_assignment='random'):
        self.classes = [KMeansClass(None, i) for i in range(self.classes_amount)]
        if labels is None:
            labels = np.zeros(len(element_set))
        self.labels = labels
        self.initialize_centroids(centroid_assignment, element_set, labels)

        while not self.stationary:
            self.update_state()

    def predict_element(self, element):
        distances = {}
        for i in range(len(self.classes)):
            distances[i] = self.classes[i].get_distance_to_class(KMeansElement(-1, element, -1, -1))
        class_index = min(distances, key=distances.get)
        return self.classes[class_index].get_most_representative_label()

    def predict(self, elements):
        predictions = []
        for element in elements:
            predictions.append(self.predict_element(element))
        return np.array(predictions)

    def plot_result(self, vertical_rot, horizontal_rot, labels):
        fig = pyplot.figure()
        ax = Axes3D(fig)
        colors = ['blue', 'red', 'black']
        color = 0
        for _class in self.classes:
            for element in _class.get_elements():
                elem = element.get_element()
                ax.scatter(elem[0], elem[1], elem[2], c=colors[color])
            color += 1
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        ax.set_zlabel(labels[2])
        ax.view_init(horizontal_rot, vertical_rot)
        plt.show()
