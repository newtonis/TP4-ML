import pandas as pd
import math

content = pd.read_csv("acath.csv")
content = content.to_dict("records")


def distEuclidia(elA, elB):
    elAx, elAy, elAz = elA
    elBx, elBy, elBz = elB

    return (elAz - elBz)**2 + (elAy - elBy)**2 + (elAx - elBx)**2


def max_distance(groupA, groupB):
    max_dist = 0
    for elementA in groupA:
        for elementB in groupB:
            max_dist = max(max_dist, distEuclidia(elementA, elementB))

    return max_dist


def min_distance(groupA, groupB):
    min_dist = 1000000
    for elementA in groupA:
        for elementB in groupB:
            min_dist = min(min_dist, distEuclidia(elementA, elementB))

    return min_dist


def average(groupA, groupB):
    pass


def centroid(groupA, groupB):
    pass


def agrupacion_jerarquica(groupA, groupB):
    pass


def get_points(content):
    full_values = []
    for item in content:
        x, y, z = item['age'], item['cad.dur'], item['choleste']
        if not math.isnan(x) and not math.isnan(y) and not math.isnan(z):
            full_values.append([x, y, z])

    return full_values


full_values = get_points(content)

print("amount of values = ", len(full_values))
print(full_values)
