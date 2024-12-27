import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# В этом задании дан датасет с синтетическими (специально сгенерированными) данными.

# Легенда
# Учёный решил провести кластеризацию некоторого множества звёзд по их расположению на карте звёздного неба.
# Кластер звёзд – это набор звёзд (точек) на карте, лежащий внутри круга радиусом R.
# Каждая звезда принадлежит ровно одному кластеру. Под расстоянием понимается расстояние Евклида.

# Описание входных данных
# В файле хранятся данные о звёздах трёх кластеров, R=3 для каждого кластера.
# В каждой строке записана информация о расположении на карте одной звезды: сначала координата x, затем координата y.
# Значения даны в условных единицах.

# Задание
# Реализуйте алгоритм k-means. С помощью реализованного алгоритма решите задачу кластеризации, распределив заданные точки по 3 кластерам.
# Для каждого кластера определите количество точек в нём и координаты центра (центроида).

def kMeans(data, k, tolerance = 1e-4):
    centroids = initializeCentroids(data, k)
    while True:
        clusters = assignPointsToClusters(data, centroids)
        newCentroids = calculateNewCentroids(clusters, centroids)
        if hasConverged(newCentroids, centroids, tolerance):
            break
        centroids = newCentroids
    return centroids, clusters

def initializeCentroids(data, k):
    return data[np.random.choice(data.shape[0], k, replace=False)]

def assignPointsToClusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        closestCentroid = np.argmin(distances)
        clusters[closestCentroid].append(point)
    return clusters

def calculateNewCentroids(clusters, oldCentroids):
    return np.array([
        np.mean(cluster, axis = 0) if cluster else oldCentroids[i]
        for i, cluster in enumerate(clusters)
    ])

def hasConverged(newCentroids, oldCentroids, tolerance = 1e-4):
    return np.max(np.abs(newCentroids - oldCentroids)) < tolerance

def showScatterPlot(centroids, clusters):
    colors = ['r', 'g', 'b']
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], c = colors[i], label = f'Кластер {i + 1}')

    centroids = np.array(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], c = 'k', marker = 'x', label = 'Центроида')
    plt.title('K-Means кластеризация')
    plt.legend()
    plt.show()

# Решение задачи
dataStars = pd.read_csv('res/data8lab.csv', delimiter = ';', decimal = ',').values

centroids, clusters = kMeans(dataStars, k = 3)

for i in range(3):
    print(f"{i + 1} кластер: ")
    print(f"Количество точек в кластере: {len(clusters[i])}")
    print(f"Координаты центроида (x, y): ({centroids[i][0]:.2f}, {centroids[i][1]:.2f})")

showScatterPlot(centroids, clusters)