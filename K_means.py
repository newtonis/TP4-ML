import random
import numpy as np


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
        self.elements.remove(element)

    def get_centroid(self):
        return self.centroid

    def get_w(self):
        w = 0
        for i in range(self.get_elements_amount()):
            for j in range(i+1, self.get_elements_amount()):
                w = (self.elements[i] - self.elements[j]) ** 2
        w /= self.get_elements_amount()
        return w

    def get_class_value(self):
        return self.class_value

    def get_distance_to_class(self, element):
        return np.linalg.norm(self.get_centroid() - element)

    def calculate_centroid(self):
        new_centroid = 0
        for element in self.elements:
            new_centroid += element
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

    def random_assignment(self, element_set, labels):
        index = 0
        for element in element_set:
            class_value = random.randint(0, self.classes_amount-1)
            self.classes[class_value].add_element(element)
            self.elements.append(KMeansElement(class_value, element, index, labels[index]))
            index += 1

    def calculate_centroids(self):
        for _class in self.classes:
            _class.calculate_centroid()

    def get_closest_class(self, k_means_element):
        distances = {}
        for _class in self.classes:
            d = _class.get_distance_to_class(k_means_element.get_element())
            distances[_class.get_class_value()] = d
        closest_class = min(distances, key=distances.get)
        return closest_class

    def update_state(self):
        has_changes = False
        self.calculate_centroids()
        for k_means_element in self.elements:
            closest_class = self.get_closest_class(k_means_element)
            if closest_class != k_means_element.get_class():
                has_changes = True
                self.classes[k_means_element.get_class()].remove_element(k_means_element.get_element())
                self.classes[closest_class].add_element(k_means_element.get_element())
                k_means_element.change_class(closest_class)
        self.stationary = not has_changes

    def fit(self, element_set, labels=None):
        self.classes = [KMeansClass(None, i) for i in range(self.classes_amount)]
        if labels is None:
            labels = np.zeros(len(element_set))
        self.labels = labels
        self.random_assignment(element_set, labels)

        while not self.stationary:
            self.update_state()

    def print_result(self, y):
        print("id       class        label")
        for i in range(len(self.elements)):
            print(self.elements[i].get_id(), "     ", self.elements[i].get_class(), "        ", y[i])
