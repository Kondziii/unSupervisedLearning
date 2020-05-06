import math
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plot
from matplotlib import animation

x1 = 0
x2 = 40
y1 = 0
y2 = 40
fig = plot.figure()
ax1 = fig.add_subplot(1, 1, 1)
# kolory dla klastrow maksymalnie 25
plotColors = ['blue', 'orange', 'green', 'red', 'purple',
              'brown', 'pink', 'gray', 'olive', 'cyan',
              'black', 'yellow', 'teal', 'fuchsia', 'lightblue',
              'darkred', 'lime', 'aquamarine', 'cornflowerblue', 'hotpink',
              'maroon', 'palegreen', 'lightyellow', 'darkgreen', 'darkblue']


class Kmeans(object):
    def __init__(self, points, animate=False, nClasters=5, maxIterations=100, availableDifference=0.001, repeats=5):
        self.points = points
        self.nClasters = nClasters
        self.availableDiff = availableDifference
        self.max_iterations = maxIterations
        self.repeats = repeats
        self.centroids = np.zeros((self.nClasters, self.points.shape[1]))
        self.previousCentroids = np.zeros((self.nClasters, self.points.shape[1]))
        self.classes = {}
        self.isStabile = np.zeros(self.nClasters)
        self.WCSS = 0
        if animate:
            self.initCentroids()
            self.clasify()
            self.stabile = False

    def initCentroids(self):
        for i in range(self.nClasters):
            self.centroids[i][0] = random.uniform(x1, x2)
            self.centroids[i][1] = random.uniform(y1, y2)
        self.isStabile[:] = False

    def clasify(self):
        for c in range(self.nClasters):
            self.classes[c] = []
        for point in self.points:
            # obliczamy dystans danego punktu od wszystkich centroidów
            distances = np.zeros(self.nClasters)
            for j in range(self.nClasters):
                for i in range(self.points.shape[1]):
                    distances[j] += (point[i] - self.centroids[j][i]) * (point[i] - self.centroids[j][i])
                distances[j] = math.sqrt(distances[j])
            # wybranie minimalnej odleglosci od klastra
            classOrder = np.argmin(distances)
            # klasyfikacja punktu
            self.classes[classOrder].append(point)

    def update(self, iteration):
        # aktualizacja polozenia centroidow - obliczenie srednich
        for c in self.classes:
            if len(self.classes[c]) != 0:
                self.centroids[c] = np.average(self.classes[c], axis=0)

        # porownujemy wartosci nowych i starych centroidow
        for c in range(self.nClasters):
            diffX = abs(self.centroids[c][0] - self.previousCentroids[c][0])
            diffY = abs(self.centroids[c][1] - self.previousCentroids[c][1])

            if diffX < self.availableDiff and diffY < self.availableDiff:
                self.isStabile[c] = True
        # zachowanie poprzedniego polozenia centroidow
        self.previousCentroids = np.array(self.centroids)
        # klasyfikacja punktów do zaktualizowanych centroidow
        self.clasify()

    def train(self, graph=True):
        error = float('inf')
        for repeat in range(self.repeats):
            # wylosowanie poczatkowych centroidow spoza zbioru danych
            self.initCentroids()
            # poczatkowa klasyfikacja
            self.clasify()
            # zdefiniowanie wektora przechowujacego poprzednie centroidy
            self.previousCentroids = np.zeros((self.nClasters, self.points.shape[1]))
            numberOfiterations = 0

            for iteration in range(self.max_iterations):
                self.update(iteration)

                # sprawdzamy czy wszystkie centroidy sa stabilne jesli tak to przerywamy iteracje
                lic = 0
                for i in range(self.nClasters):
                    if self.isStabile[i]:
                        lic += 1
                if lic == self.nClasters:
                    numberOfiterations = iteration + 1
                    break
                numberOfiterations += 1
            # obliczenie bledu
            repeatError = self.calculateWCSS()
            if repeatError < error:
                error = repeatError
                finalCentroids = np.array(self.centroids)
                finalClasses = dict(self.classes)
        self.WCSS = error
        self.centroids = finalCentroids
        self.classes = finalClasses
        # wyswietlenie wykresu klasyfikacji
        if graph:
            self.drawPlot(numberOfiterations)

    def drawPlot(self, numberOfIterations):
        for c in self.classes:
            for point in self.classes[c]:
                plot.scatter(point[0], point[1], color=plotColors[c])
            plot.scatter(self.centroids[c][0], self.centroids[c][1], color=plotColors[c], s=110, marker='s')
        plot.title('Chart for ' + self.nClasters.__str__() + ' clasters' +
                   '\nNumber of iterations: ' + numberOfIterations.__str__() + ', WCSS = ' + self.WCSS.__str__())
        # plot.savefig(self.nClasters.__str__()+'x.png')
        plot.show()

    def calculateWCSS(self):
        WCSS = 0  # within cluster sum of squares
        for c in self.classes:
            for point in self.classes[c]:
                for i in range(self.points.shape[1]):
                    WCSS += math.sqrt((point[i] - self.centroids[c][i]) * (point[i] - self.centroids[c][i]))
        return WCSS

    def initPlot(self):
        text = 'Number of iterations: 0\n Centroids stability: '
        if not self.stabile:
            text += 'unstable'
        else:
            text += 'stable'
        ax1.clear()
        ax1.set_xlim([x1 - 1, x2 + 1])
        ax1.set_ylim([y1 - 1, y2 + 1])
        ax1.title.set_text(text)
        ax1.plot(self.points[:, 0], self.points[:, 1], 'ro')
        for c in range(self.nClasters):
            ax1.scatter(self.centroids[c][0], self.centroids[c][1], color=plotColors[c], s=150, marker='s')

    def animate(self, t):
        if t == 0:
            self.initPlot()
        else:
            ax1.clear()
            self.update(t - 1)
            lic = 0
            for i in range(self.nClasters):
                if self.isStabile[i]:
                    lic += 1
            if lic == self.nClasters:
                self.stabile = True
            text = 'Number of iterations: ' + str(t) + '\n Centroids stability: '
            if not self.stabile:
                text += 'unstable'
            else:
                text += 'stable'
            ax1.clear()
            ax1.set_xlim([x1 - 1, x2 + 1])
            ax1.set_ylim([y1 - 1, y2 + 1])
            ax1.title.set_text(text)
            for c in self.classes:
                for point in self.classes[c]:
                    plot.scatter(point[0], point[1], color=plotColors[c])
            for c in range(self.nClasters):
                ax1.scatter(self.centroids[c][0], self.centroids[c][1], color=plotColors[c], s=150, marker='s')
            if self.stabile:
                ani._stop()


# funkcja umozliwiająca wylosowac punkty i zapisac je do pliku randPoints.txt
def randPoints(numberOfPoints):
    points = np.zeros((numberOfPoints, 2))
    for i in range(numberOfPoints):
        points[i][0] = random.uniform(x1, x2)
        points[i][1] = random.uniform(y1, y2)
        file = open("randPoints.txt", "w")
        for i in range(numberOfPoints):
            file.write(str(points[i][0]) + "," + str(points[i][1]) + '\n')
        file.close()
    return points


# losowanie punktow
points = randPoints(100)
# odczytanie punktow z pliku
# data = np.array(pd.read_csv('randPoints.txt', names=['x', 'y'], header=None))
# kmeans1 = Kmeans(data)
# kmeans1.train()

######## animacja
data = np.array(pd.read_csv('randPoints.txt', names=['x', 'y'], header=None))
kmeans2 = Kmeans(data, True)
ani = animation.FuncAnimation(fig, kmeans2.animate, frames=20, interval=2000)
plot.show()


# error = []
# clusters = []
# for i in range(50):
#     kmeans = Kmeans(x, False, i + 1, 100, 0.001, 20)
#     kmeans.train(False)
#     error.append(kmeans.WCSS)
#     clusters.append(i+1)
#
# plot.plot(clusters, error, 'blue')
# plot.xlabel('Number of clasters')
# plot.ylabel('Error')
# plot.savefig('error2.png')
# plot.show()
