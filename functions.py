import numpy as np
import pandas as pd

MEAN = 0
STD = 1


# η = 0,1 ∗(1 −epoca/max ctd epocas
class LearningRateFunction:
    def __init__(self, max_learning_rate, iterations):
        self.max_learning_rate = max_learning_rate
        self.max_iterations = iterations
        self.count = 0

    def get_next_learning_rate(self):
        if self.count == 0:
            return self.max_learning_rate
        else:
            return self.max_learning_rate * (1 - self.count / self.max_iterations)
    #def get_next_learning_rate(self):
    #    self.count += 1
    #    if self.learning_rate / (self.count / 5) <= 1:
    #        self.learning_rate = self.learning_rate / (self.count / 5)
    #    else:
    #        self.learning_rate = 1
    #    return self.learning_rate


# R = (max ctd epocas −epoca) ∗Rinicial /max ctd epocas )
class RadiusFunction:
    def __init__(self, max_radius, iterations):
        self.max_radius = max_radius
        self.max_iterations = iterations
        self.count = 0

    def get_next_radius(self):
        return (self.max_iterations - self.count) * self.max_radius / self.max_iterations

    #def __init__(self, network_size, sorting_iterations_number):
    #    self.init_radius = network_size / 2
    #    self.count = 0
    #    self.sorting_iterations_number = sorting_iterations_number

    #def get_next_radius(self):
    #    self.count += 1
    #    return self.init_radius * (self.sorting_iterations_number - self.count) / self.sorting_iterations_number


def calculate_stats(df, column):
    mean = df[column].mean()
    std = df[column].std()
    return [mean, std]


def standarize(data_set):
    cols = data_set.columns
    stats = [calculate_stats(data_set, column) for column in cols]
    data = data_set.to_numpy()
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = (data[i][j] - stats[j][MEAN])/stats[j][STD]
    return data