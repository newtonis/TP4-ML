import pandas as pd
import math

content = pd.read_csv("acath.csv")
content = content.to_dict("records")


class Node:
    def __init__(self, left, right, height, value):
        self.left = left
        self.right = right
        self.height = height
        self.value = value

    def visualize_structure(self):
        print("node data: ")

        if self.value == None:
            print("value: None")
        else:
            print("value:", self.value)

        if self.value == None:
            print("height: None")
        else:
            print("height: %f" % self.height)

        if self.left != None:
            print("left Node")
            self.left.visualize_structure()

        if self.right != None:
            print("right Node")
            self.right.visualize_structure()

# celsius
# nexo => hacen lo mismo, la diferencia de tasas no es tanto.
# Es verdaderamente seguro por como es el tradeo con margen.
# Es imposible que la persona que tomo plata prestada no la devuelva.
# binance p2p, clasico. Atrapados por gob totalitario, para fondear binance es con p2p.
# Hace una semana pusieron un impuesto.


def computeNodesDistances(nodeA, nodeB, type):
    nodesA = getAllLeafNodes(nodeA)
    nodesB = getAllLeafNodes(nodeB)

    if type == 'max_distance':
        return max_distance(nodesA, nodesB)
    elif type == 'min_distance':
        return min_distance(nodesA, nodesB)
    elif type == 'avg_distance':
        return average(nodesA, nodesB)
    elif type == 'centroid':
        return centroid(nodesA, nodesB)


def getAllLeafNodes(node):
    leaf_nodes = []
    nodes = [node]
    while len(nodes) > 0:
        if nodes[0].value != None:
            leaf_nodes.append(nodes[0])
        else:
            if nodes[0].left.value == None:
                nodes.append(nodes[0].left)
            else:
                leaf_nodes.append(nodes[0].left)

            if nodes[0].right.value == None:
                nodes.append(nodes[0].right)
            else:
                leaf_nodes.append(nodes[0].right)

        del nodes[0]

    values = []
    for node in leaf_nodes:
        values.append(node.value)

    return values


def createLeafNode(value):
    return Node(None, None, 0, value)


def createNode(left, right, height):
    return Node(left, right, height, None)


def distEuclidia(elA, elB):
    elAx, elAy, elAz, ans = elA
    elBx, elBy, elBz, ans = elB

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
    sum_dist = 0
    elements_sum = 0
    for elementA in groupA:
        for elementB in groupB:
            sum_dist += distEuclidia(elementA, elementB)
            elements_sum += 1

    return sum_dist / elements_sum


def centroid(groupA, groupB):
    centroidA = [0, 0, 0, 0]
    countA = 0

    for elementA in groupA:
        x, y, z, ans = elementA
        centroidA[0] += x
        centroidA[1] += y
        centroidA[2] += z
        countA += 1

    centroidA[0] /= countA
    centroidA[1] /= countA
    centroidA[2] /= countA

    centroidB = [0, 0, 0, 0]
    countB = 0

    for elementB in groupB:
        x, y, z, ans = elementB
        centroidB[0] += x
        centroidB[1] += y
        centroidB[2] += z
        countB += 1

    centroidB[0] /= countB
    centroidB[1] /= countB
    centroidB[2] /= countB

    centroidAns = distEuclidia(centroidA, centroidB)

    return centroidAns


def agrupacion_jerarquica(values, type):
    nodes = []

    for i in range(len(values)):
        nodes.append(createLeafNode(values[i]))

    while len(nodes) > 1:
        #print("iteration: ")

        for item in nodes:
            get_leaf = getAllLeafNodes(item)
            items = ""
            for item_b in get_leaf:
                items += str(item_b) + " "
            #print(items)

        sel_x = None
        sel_y = None
        min_dist = 100000000000
        for x in range(len(nodes)):
            for y in range(x+1, len(nodes)):
                computed_distance = computeNodesDistances(nodes[x], nodes[y], type)
                if computed_distance < min_dist:
                    min_dist = computed_distance
                    sel_x = x
                    sel_y = y

        nodes[sel_x] = createNode(nodes[sel_x], nodes[sel_y], min_dist)
        del nodes[sel_y]

    return nodes[0]


def get_points(content):
    full_values = []
    for item in content:
        x, y, z = item['age'], item['cad.dur'], item['choleste']
        ans = item['sigdz']

        if not math.isnan(x) and not math.isnan(y) and not math.isnan(z):
            #print(x, y, z, ans)
            full_values.append([x, y, z, ans])

    return full_values


full_values = get_points(content[:1000])

versions = ['max_distance', 'min_distance', 'centroid', 'avg_distance']

for version in versions:
    print("step = ", version)
    answer = agrupacion_jerarquica(full_values, version)
    #answer.visualize_structure()

    valuesA = getAllLeafNodes(answer.left)
    valuesB = getAllLeafNodes(answer.right)

    distA = [0, 0]
    distB = [0, 0]

    for value in valuesA:
        if value[3] == 0:
            distA[0] += 1
        else:
            distA[1] += 1

    for value in valuesB:
        if value[3] == 0:
            distB[0] += 1
        else:
            distB[1] += 1

    print(distA, distB)

    # we need accuracy

    print("amount of values = ", len(full_values))
    #print(full_values)
