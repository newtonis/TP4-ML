import math
import random
import sys

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from random import shuffle



class KohonenElement:
    def __init__(self, class_value, weights):
        self.class_value = class_value
        self.weights = weights

    def get_class_value(self):
        return self.class_value

    def get_weights(self):
        return self.weights


class KohonenNetwork:
    def __init__(self, network_size, input_size, network_type):
        self.network = [[KohonenNeuron(input_size, j, i) for j in range(network_size)] for i in range(network_size)]
        self.input_size = input_size
        self.network_size = network_size
        self.type = network_type
        self.training_set = None

    def create_kohonen_elements(self, data, labels):
        training_set = []
        for i in range(len(data)):
            training_set.append(KohonenElement(labels[i], data[i]))
        self.training_set = training_set

    # si hay datos no numericos que identifican, se asume que es la primera columna
    def initialize(self, training_set, labels):
        self.create_kohonen_elements(training_set, labels)
        for i in range(self.network_size):
            for j in range(self.network_size):
                self.network[i][j].weights = self.training_set[random.randint(0, len(self.training_set)-1)].get_weights()

    def train(self, iterations, learning_rate_function, radius_function):
        for i in range(iterations):
            self.clear_marks()
            shuffle(self.training_set)
            for training_example in self.training_set:
                winner_neuron = self.get_representative(training_example)
                vicinity = self.get_vicinity(winner_neuron, radius_function.get_next_radius())  # por ahora radio es cte
                self.update(vicinity, training_example, learning_rate_function.get_next_learning_rate())

    def get_representative(self, input_example):
        min_distance = sys.maxsize
        winner = None
        for i in range(self.network_size):
            for j in range(self.network_size):
                curr_distance = self.network[i][j].get_euclidean_distance(input_example.get_weights())
                if curr_distance < min_distance:
                    min_distance = curr_distance
                    winner = self.network[i][j]
        winner.times_chosen += 1
        winner.add_representative(input_example.get_class_value())
        return winner

    def get_representative_for_predict(self, input_example):
        min_distance = sys.maxsize
        winner = None
        for i in range(self.network_size):
            for j in range(self.network_size):
                curr_distance = self.network[i][j].get_euclidean_distance(input_example)
                if curr_distance < min_distance:
                    min_distance = curr_distance
                    winner = self.network[i][j]
        winner.times_chosen += 1
        return winner

    def clear_marks(self):
        for i in range(self.network_size):
            for j in range(self.network_size):
                self.network[i][j].times_chosen = 0
                self.network[i][j].clear_representatives()

    def update(self, vicinity, training_example, learning_rate):
        for neuron in vicinity:
            delta = learning_rate * np.subtract(training_example.get_weights(), neuron.weights) * (1 / neuron.distance_from_winner)
            neuron.weights = np.add(neuron.weights, delta)
            neuron.distance_from_winner = sys.maxsize

    def get_vicinity(self, winner_neuron, radius):
        winner_neuron.distance_from_winner = 1
        vicinity = [winner_neuron]
        for i in range(self.network_size):
            for j in range(self.network_size):
                if i != winner_neuron.y_coord or j != winner_neuron.x_coord:
                    self.network[i][j].distance_from_winner = winner_neuron.get_euclidean_grid_distance(self.network[i][j])
                    if self.network[i][j].distance_from_winner < radius:
                        vicinity.append(self.network[i][j])
        return vicinity

    def predict(self, x_test, y_test):
        results = []
        for i in range(len(x_test)):
            winner = self.get_representative_for_predict(x_test[i])
            results.append([winner.get_most_represented_class(), y_test[i]])
        return results

    def build_U_matrix(self):
        feature_map = [[self.get_average_neighbour_distance(self.network[i][j]) for j in range(self.network_size)] for i in range(self.network_size)]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(' ', ['white', 'grey', 'black'])
        plt.imshow(feature_map, cmap=cmap)
        plt.title("Matriz U")
        plt.colorbar()
        plt.axis('off')
        plt.show()

    def get_average_neighbour_distance(self, neuron):
        x_coord = neuron.x_coord
        y_coord = neuron.y_coord
        distances = 0
        distances_sum = 0

        # upper distance
        if y_coord-1 >= 0:
            distances_sum += self.network[y_coord-1][x_coord].get_euclidean_distance(neuron.weights)
            distances += 1
        # lower distance
        if y_coord+1 < self.network_size:
            distances_sum += self.network[y_coord+1][x_coord].get_euclidean_distance(neuron.weights)
            distances += 1
        # get left distance
        if x_coord-1 >= 0:
            distances_sum += self.network[y_coord][x_coord-1].get_euclidean_distance(neuron.weights)
            distances += 1
        # get right distance
        if x_coord+1 < self.network_size:
            distances_sum += self.network[y_coord][x_coord+1].get_euclidean_distance(neuron.weights)
            distances += 1

        if self.type == "hexagonal":
            # get right upper side
            if x_coord+1 < self.network_size and y_coord-1 > 0:
                distances_sum += self.network[y_coord-1][x_coord+1].get_euclidean_distance(neuron.weights)
                distances += 1
            # get right lower side
            if x_coord+1 < self.network_size and y_coord+1 < self.network_size:
                distances_sum += self.network[y_coord+1][x_coord+1].get_euclidean_distance(neuron.weights)
                distances += 1
        neuron.avg_distance = distances_sum / distances
        return neuron.avg_distance

    def plot_network_features(self, variable_names):
        for k in range(self.input_size):
            feature_map = [[self.network[i][j].weights[k] for j in range(self.network_size)] for i in range(self.network_size)]
            plt.title(variable_names[k])
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(' ', ['blue', 'yellow', 'red'])
            plt.imshow(feature_map, cmap=cmap)
            plt.colorbar()
            plt.axis('off')
            plt.show()

    def plot_clusters(self):
        colors = ['blue', 'red', 'black']
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i in range(self.network_size):
            for j in range(self.network_size):
                class_value = self.network[i][j].get_most_represented_class()
                if class_value is None:
                    c = 'black'
                else:
                    c = colors[class_value]
                    ax.annotate('%s' % str(class_value), xy=(j, i), xytext=(10, 20 - 10),
                                textcoords='offset points')  # xy es la label?
                ax.scatter(j, i, c=c, s=70)

        plt.xlim(0, self.network_size-1)
        plt.ylim(0, self.network_size-1)
        plt.xticks([i for i in range(self.network_size)])
        plt.yticks([j for j in range(self.network_size)])
        plt.gca().invert_yaxis()
        plt.grid()
        plt.show()

    def plot_network_hit_map(self):
        hit_map = [[self.network[i][j].times_chosen for j in range(self.network_size)] for i in range(self.network_size)]
        plt.title("cantidad de registros que van a cada nodo")
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(' ', ['blue', 'yellow', 'red'])
        plt.imshow(hit_map, cmap=cmap)
        plt.colorbar()
        plt.axis('off')
        plt.show()


class KohonenNeuron:
    def __init__(self, weights_amount, x_coord, y_coord):
        self.weights = np.zeros(weights_amount)
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.avg_distance = sys.maxsize
        self.times_chosen = 0
        self.distance_from_winner = sys.maxsize
        self.represents = {}

    def clear_representatives(self):
        self.represents.clear()

    def add_representative(self, class_value):
        if class_value in self.represents:
            self.represents[class_value] += 1
        else:
            self.represents[class_value] = 1

    def get_most_represented_class(self):
        if not self.represents:
            return None
        most_represented = max(self.represents, key=self.represents.get)
        return most_represented

    def get_euclidean_distance(self, coord):
        return np.linalg.norm(self.weights - coord)

    def get_manhattan_grid_distance(self, neuron):
        d = abs(self.x_coord - neuron.x_coord) + abs(self.y_coord - neuron.y_coord)
        return d

    def get_euclidean_grid_distance(self, neuron):
        d = math.sqrt(((self.x_coord - neuron.x_coord) ** 2) + ((self.y_coord - neuron.y_coord) ** 2))
        return d