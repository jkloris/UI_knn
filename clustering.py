import numpy as np
import random
from dataclasses import  dataclass
# from binarytree import Node
MAPSIZE = [-5000, 5000]

# Red = 0
# Green = 1
# Blue = 2
# Purple = 3

@dataclass(order=True)
class Node:
    value: np.ndarray
    left: ["Node"] = None
    right: ["Node"] = None


class KDTree:

    def __init__(self, trainingSet):
        self.trainingSet = trainingSet
        self.tree = self.buildTree()

    def buildTree(self):
        self.trainingSet = self.trainingSet[self.trainingSet[:,0].argsort()]
        # print(trainingSet)
        med = int(len(self.trainingSet) / 2)
        leftSet = self.trainingSet[:med-1]
        rightSet = self.trainingSet[med:]

        root = Node(self.trainingSet[med])

        # rightSet by mal byt vacsi
        for i, value in enumerate(rightSet):
            root = self.addNode(root, value, 0)
            if i < len(leftSet):
                root = self.addNode(root, leftSet[i], 0)

        return root

    # depth 0 -> 1 -> 2 -> ...; na kazdej parnej sa rozhoduje podla value[0], neparnej value[1]
    def addNode(self, node, value, depth):

        if not depth % 2:
            if node.value[0] > value[0]:
                if node.right:
                    node.right = self.addNode(node.right, value, depth+1)
                    return node
                else:
                    node.right = Node(value)
                    return node
            elif node.value[0] < value[0]:
                if node.left:
                    node.left = self.addNode(node.left, value, depth+1)
                    return node
                else:
                    node.left = Node(value)
                    return node
            else:
                if node.value[1] < value[1]:
                    if node.left:
                        node.left = self.addNode(node.left, value, depth + 1)
                        return node
                    else:
                        node.left = Node(value)
                        return node
                elif node.value[1] >= value[1]:
                    if node.right:
                        node.right = self.addNode(node.right, value, depth + 1)
                        return node
                    else:
                        node.right = Node(value)
                        return node
        else:
            if node.value[0] > value[0]:
                if node.left:
                    node.left = self.addNode(node.left, value, depth + 1)
                    return node
                else:
                    node.left = Node(value)
                    return node
            elif node.value[0] < value[0]:
                if node.right:
                    node.right = self.addNode(node.right, value, depth + 1)
                    return node
                else:
                    node.right = Node(value)
                    return node
            else:
                if node.value[1] < value[1]:
                    if node.right:
                        node.right = self.addNode(node.right, value, depth + 1)
                        return node
                    else:
                        node.right = Node(value)
                        return node
                elif node.value[1] >= value[1]:
                    if node.left:
                        node.left = self.addNode(node.left, value, depth + 1)
                        return node
                    else:
                        node.left = Node(value)
        return node



def init():
    trainingSet = np.array(([-4500, -4400, 0], [-4100, -3000, 0], [-1800, -2400, 0], [-2500, -3400, 0], [-2000, -1400, 0],
                  [4500, -4400, 1], [4100, -3000, 1], [1800, -2400, 1], [2500, -3400, 1], [2000, -1400, 1],
                  [-4500, 4400, 2], [-4100, 3000, 2], [-1800, 2400, 2], [-2500, 3400, 2], [-2000, 1400, 2],
                  [4500, 4400, 3], [4100, 3000, 3], [1800, 2400, 3], [2500, 3400, 3], [2000, 1400, 3]), dtype=np.int16)
    return trainingSet

# flag: znacka farby 0-3 (R,G,B,P)
#TODO odstranit duplikacie
def generatePoint(flag):
    r = random.randint(0,100)
    if not r:
        point = np.random.randint(MAPSIZE[0], MAPSIZE[1], size=(2))
        return point
    else:
        if flag == 0:
            point = np.random.randint(MAPSIZE[0], 500, size=(2))
            return point
        if flag == 1:
            point = np.array((random.randint(-500,MAPSIZE[1]), random.randint(MAPSIZE[0],500)))
            return point
        if flag == 2:
            point = np.array((random.randint(MAPSIZE[0], 500), random.randint(-500, MAPSIZE[1])))
            return point
        if flag == 3:
            point = np.random.randint(-500, MAPSIZE[1], size=(2))
            return point


def demoClasification(trainingSet, range):

    for i in range(range):
        flag = i % 4
        p = generatePoint(flag)


def getDistance(p1, p2):
    d = np.linalg.norm(p1[:2]-p2[:2])
    return int(d)

# vrati k najblizsich
# demo, len na skusku
def demoFindNearest(trainingSet, newPoint, k):
    nearest = []
    for i, p in enumerate(trainingSet):
        d = getDistance(p, newPoint)
        if i < k:
            nearest.append([p, d])
            if i == k-1:
                nearest.sort(key=sortFunc)
            continue
        elif d < nearest[k-1][1]:
            nearest[k-1] = [p, d]
            nearest.sort(key=sortFunc)

    return nearest

# na sortovanie podla dlzky, vrati element na druhom indexe
def sortFunc(x):
    return x[1]

def checkMajority(nearests):
    count = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }

    for i in nearests:
        count[i[0][2]]+=1

    return max(count, key=count.get)





if __name__ == "__main__":
    trainingSet = init()
    k = 5


    kdtree = KDTree(trainingSet)


    for i in range(500):
        p = generatePoint(i % 4)
        nearest = demoFindNearest(trainingSet,p , k)
        p = np.append(p, checkMajority(nearest))
        # trainingSet = np.append(trainingSet, [p], axis=0)
        kdtree.addNode(kdtree.tree, p, 0)

print(trainingSet)

# notes:
# 100_000 /