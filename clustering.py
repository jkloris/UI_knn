import numpy as np
import random
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
import threading
import copy

MAPSIZE = [-5000, 5000]
SECTORSIZE = 100
# Red = 0
# Green = 1
# Blue = 2
# Purple = 3



class Sector:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.count = 0
        self.points = np.empty((0, 3))

    def addPoint(self, point):
        self.points = np.append(self.points, [point], axis=0)
        self.count+=1



@dataclass(order=True)
class Node:
    value: np.ndarray
    left: ["Node"] = None
    right: ["Node"] = None


class KDTree:

    def __init__(self, trainingSet):
        self.trainingSet = trainingSet
        self.tree = self.buildTree()
        self.knearests = None

    def buildTree(self):
        self.trainingSet = self.trainingSet[self.trainingSet[:,0].argsort()]
        # print(trainingSet)
        med = int(len(self.trainingSet) / 2)
        leftSet = self.trainingSet[:med]
        rightSet = self.trainingSet[med+1:]

        root = Node(self.trainingSet[med])

        # leftSet by mal byt vacsi
        for i, value in enumerate(leftSet):
            root = self.addNode(root, value, 0)
            if i < len(rightSet):
                root = self.addNode(root, rightSet[i], 0)

        return root

    # depth 0 -> 1 -> 2 -> ...; na kazdej parnej sa rozhoduje podla value[0], neparnej value[1]
    # zle!?.. :(
    def addNode(self, node, value, depth):

        if not depth % 2:
            if node.value[0] < value[0]:
                if node.right:
                    node.right = self.addNode(node.right, value, depth+1)
                    return node
                else:
                    node.right = Node(value)
                    return node
            elif node.value[0] > value[0]:
                if node.left:
                    node.left = self.addNode(node.left, value, depth+1)
                    return node
                else:
                    node.left = Node(value)
                    return node
            else:
                if node.value[1] > value[1]:
                    if node.left:
                        node.left = self.addNode(node.left, value, depth + 1)
                        return node
                    else:
                        node.left = Node(value)
                        return node
                elif node.value[1] <= value[1]:
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
            elif node.value[1] > value[1]:
                if node.left:
                    node.left = self.addNode(node.left, value, depth + 1)
                    return node
                else:
                    node.left = Node(value)
                    return node
            else:
                if node.value[0] > value[0]:
                    if node.left:
                        node.left = self.addNode(node.left, value, depth + 1)
                        return node
                    else:
                        node.left = Node(value)
                        return node
                elif node.value[0] <= value[0]:
                    if node.right:
                        node.right = self.addNode(node.right, value, depth + 1)
                        return node
                    else:
                        node.right = Node(value)
        return node

    def findNearestKD(self, point, node, depth, shortest):
        # Go to the leaf, in every node measure dist
        # Ak sme v liste, vypocitaj kolmu vzdialenost k rodicovi
        # Ak nie je blizsia ako najblizsia namerana, idem hore
        # Ak je blizsia, vnorim sa

        dist = getDistance(point, node.value)
        if dist < shortest:
            shortest = dist
            self.knearests = node.value

        if node.right != None and node.left != None:
            if not depth % 2:
                if point[0] > node.value[0]:
                    shortest = self.findNearestKD(point, node.right, depth + 1, shortest)
                elif point[0] < node.value[0]:
                    shortest = self.findNearestKD(point, node.left, depth + 1, shortest)
                else:
                    if point[1] > node.value[1]:
                        shortest = self.findNearestKD(point, node.right, depth + 1, shortest)
                    elif point[1] <= node.value[1]:
                        shortest = self.findNearestKD(point, node.left, depth + 1, shortest)
            elif depth % 2:
                if point[1] > node.value[1]:
                    shortest = self.findNearestKD(point, node.right, depth + 1, shortest)
                elif point[1] < node.value[1]:
                    shortest = self.findNearestKD(point, node.left, depth + 1, shortest)
                else:
                    if point[0] > node.value[0]:
                        shortest = self.findNearestKD(point, node.right, depth + 1, shortest)
                    elif point[0] <= node.value[0]:
                        shortest = self.findNearestKD(point, node.left, depth + 1, shortest)

        elif node.right == None and node.left != None:
            shortest = self.findNearestKD(point, node.left, depth + 1, shortest)

        elif node.right != None and node.left == None:
            shortest = self.findNearestKD(point, node.right, depth + 1, shortest)

        else:
            return shortest

        # cesta spat
        if not depth % 2:
            if abs(point[0] - node.value[0]) < shortest:
                if node.right != None and point[0] - node.value[0] < 0 :
                    shortest = self.findNearestKD(point, node.right, depth + 1, shortest)
                elif node.left != None and point[0] - node.value[0] >= 0 :
                    shortest = self.findNearestKD(point, node.left, depth + 1, shortest)
        else:
            if abs(point[1] - node.value[1]) < shortest:
                if node.right != None and point[1] - node.value[1] < 0:
                    shortest = self.findNearestKD(point, node.right, depth + 1, shortest)
                elif node.left != None and point[1] - node.value[1] >= 0:
                    shortest = self.findNearestKD(point, node.left, depth + 1, shortest)

        return shortest


def init():
    trainingSet = np.array(([-4500, -4400, 0], [-4100, -3000, 0], [-1800, -2400, 0], [-2500, -3400, 0], [-2000, -1400, 0],
                  [4500, -4400, 1], [4100, -3000, 1], [1800, -2400, 1], [2500, -3400, 1], [2000, -1400, 1],
                  [-4500, 4400, 2], [-4100, 3000, 2], [-1800, 2400, 2], [-2500, 3400, 2], [-2000, 1400, 2],
                  [4500, 4400, 3], [4100, 3000, 3], [1800, 2400, 3], [2500, 3400, 3], [2000, 1400, 3]), dtype=np.int16)
    return trainingSet

# @param flag : znacka farby 0-3 (R,G,B,P)
#TODO odstranit duplikacie
def generatePoint(flag, points):
    r = random.randint(0,100)
    while True:
        if not r:
            point = np.array((int(random.random()*9999)-5000, int(random.random()*9999)-5000), dtype=np.int16)
            if not any(np.equal(points, point).all(1)):
                return point
            continue
        else:
            rx = int(random.random()*5499)
            ry = int(random.random()*5499)

            if flag == 0:
                point = np.array((rx - 5000, ry - 5000), dtype=np.int16)
                if not any(np.equal(points, point).all(1)):
                    return point
                continue
            if flag == 1:
                point = np.array((rx - 500, ry - 5000), dtype=np.int16)
                if not any(np.equal(points, point).all(1)):
                    return point
                continue
            if flag == 2:
                point = np.array((rx - 5000, ry - 500), dtype=np.int16)
                if not any(np.equal(points, point).all(1)):
                    return point
                continue
            if flag == 3:
                point = np.array((rx - 500, ry - 500), dtype=np.int16)
                if not any(np.equal(points, point).all(1)):
                    return point
            continue


def createRandomPoints(size, al):
    points = al[0]
    # points = []
    for i in range(size):
        points = np.append(points, [generatePoint(i % 4, points)], axis=0)
        # points.append(generatePoint(i % 4, points))
    al[0] = points
    return points


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

# todo vyriesit ak sa rovnaju
def checkMajority(nearests):
    count = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }

    for i in nearests:
        count[i[0][2]]+=1

    maxx = max(count, key=count.get)
    for i in count.keys():
        if count[maxx] == count[i]:
            for n in nearests:
                if n[0][2] == maxx or n[0][2] == i:
                    maxx = n[0][2]
                    break


    return maxx


def createSectors():
    sectors = []
    for r in range(int(MAPSIZE[1]*2 / SECTORSIZE)):
        sectors.append([])
        for c in range(int(MAPSIZE[1]*2 / SECTORSIZE)):
            sectors[r].append(Sector(r,c))

    return sectors

def getPosOfPoint(point):
    r = int((MAPSIZE[1] + point[0]) / SECTORSIZE)
    c = int((MAPSIZE[1] + point[1]) / SECTORSIZE)
    return [r,c]

def addPointToSector(sectors, point):
    r,c = getPosOfPoint(point)
    sectors[r][c].addPoint(point)


def findKnearestInSectors(sectors, point, k, size, kdtree):
    kmagic = 1
    if k >= 15:
        if size < 3000:
            kmagic = k
        elif size < 10000:
            kmagic = 7
        else:
            kmagic = 10

    elif k >= 7:
        if size < 4000:
            kmagic = k
        elif size < 8000:
            kmagic = 4
        else:
            kmagic = 5

    elif k >= 3:
        kmagic = k


    magic = int(math.sqrt(MAPSIZE[1] * 2 / size)) * kmagic
    magic = magic if magic > 1 else  kmagic
    susedia = []


    dist = kdtree.findNearestKD(point, kdtree.tree, 0, 15000)
    susedia.append([kdtree.knearests, dist])

    if k == 1:
        return susedia

    distSector = int(dist / SECTORSIZE) + magic  #malo v skorych fazach

    # distSector = magic * k
    rPoint, cPoint = getPosOfPoint(point)

    r1 = 0 if rPoint - distSector <= 0 else rPoint - distSector
    r2 = int(MAPSIZE[1]*2 / SECTORSIZE) - 1 if rPoint + distSector >= int(MAPSIZE[1]*2 / SECTORSIZE) else rPoint + distSector

    c1 = 0 if cPoint - distSector <= 0 else cPoint - distSector
    c2 = int(MAPSIZE[1] * 2 / SECTORSIZE) - 1 if cPoint + distSector >= int(MAPSIZE[1] * 2 / SECTORSIZE) else cPoint + distSector


    for r in range(r1,r2,1):
        for c in range(c1,c2,1):
            if sectors[r][c].count == 0:
                continue
            for s in sectors[r][c].points:
                d = getDistance(point, s)

                if len(susedia) < k:
                    susedia.append([s, d])
                    if len(susedia) == k:
                        susedia.sort(key=sortFunc)
                    continue
                elif d < susedia[-1][1]:
                    susedia[k - 1] = [s, d]
                    susedia.sort(key=sortFunc)

    if len(susedia) != k:
        print(len(susedia))

    return susedia


def clasify(point, k, kdtree, sectors, size):
    susedia = findKnearestInSectors(sectors, point, k, size, kdtree) # todo zmenit size
    newClass = checkMajority(susedia)
    point = np.append(point, newClass)
    r, c = getPosOfPoint(point)
    try:
        sectors[r][c].addPoint(point)
    except:
        print(r, c, "###########")

    kdtree.addNode(kdtree.tree, point, 0)
    kdtree.trainingSet = np.append(kdtree.trainingSet, [point], axis=0)
    return newClass

def fillSectors(sectors, trainingSet):
    for p in trainingSet:
        addPointToSector(sectors, p)


def threadClasify(k, rangeN, points):
    trainingSet = init()
    sectors = createSectors()
    fillSectors(sectors, trainingSet)
    newkdtree = KDTree(trainingSet)

    eqClassCount = 0
    for i in range(rangeN):
        p = points[i]
        newClass = clasify(p, k, newkdtree, sectors, i+20)

        if newClass == i % 4:
            eqClassCount += 1

        if i % int(rangeN / 100) == 0:
            print(f"#", end="")

    print(f"\nUspesnost pre k == {k}: {eqClassCount / rangeN * 100}%")

    x = newkdtree.trainingSet[..., 0]
    y = newkdtree.trainingSet[..., 1]
    colors = newkdtree.trainingSet[..., 2]

    # rgbp = ListedColormap(["red", "green", "blue", "purple"])
    # plt.figure(figsize=(4, 4))
    # plt.scatter(x, y, s=15, c=colors, cmap=rgbp)

    return [x,y, colors]

if __name__ == "__main__":

    rangeN = 20000

    points = np.empty((0, 2))
    al = [points]
    thset = threading.Thread(target=createRandomPoints, args=(rangeN, al))
    thset.start()

    print("1...Zvolenie si vlastneho k\n2...Testovanie na vsetkych k (1, 3, 7, 15)")
    cmd = int(input())
    if cmd == 1:
        print("Zvol is hodnotu k:")
        k = int(input())


    thset.join()
    points = al[0]

    # nakreslenie mierky
    for i in range(100):
        if i % 10 == 0:
            print(f"{i}", end="")
        elif i % 10 != 9:
            print("-", end="")
    print("100")

    if cmd == 1:
        x1, y1, colors1 = threadClasify(k, rangeN, np.copy(points))
        rgbp = ListedColormap(["red", "green", "blue", "purple"])
        plt.figure(figsize=(4, 4))
        plt.scatter(x1, y1, s=15, c=colors1, cmap=rgbp)
        plt.show()
    elif cmd == 2:




        x1, y1, colors1 = threadClasify(1, rangeN, np.copy(points))
        # x3, y3, colors3 = threadClasify(7, rangeN, np.copy(points))
        # x4, y4, colors4 = threadClasify(15, rangeN, np.copy(points))
        x2, y2, colors2 = threadClasify(3, rangeN, np.copy(points))
        # threadClasify(7, rangeN, np.copy(points))

        rgbp = ListedColormap(["red", "green", "blue", "purple"])
        plt.figure(figsize=(4, 4))
        plt.scatter(x1, y1, s=15, c=colors1, cmap=rgbp)

        plt.figure(figsize=(4, 4))
        plt.scatter(x2, y2, s=15, c=colors2, cmap=rgbp)
        plt.show()

        # plt.figure(figsize=(4, 4))
        # plt.scatter(x3, y3, s=15, c=colors3, cmap=rgbp)
        #
        # plt.figure(figsize=(4, 4))
        # plt.scatter(x4, y4, s=15, c=colors4, cmap=rgbp)







# notes:
# nastavit lepsie magic number
# niekde je chyba