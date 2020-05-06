import math
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plot
from matplotlib import animation

learningRateMax = 0.5
learningRateMin = 0.01
radiusMax = 1
radiusMin = 0.01
potentialMin = 0.75

fig = plot.figure()
ax1 = fig.add_subplot(1, 1, 1)

class Som(object):
    def __init__(self, inputs, nNeurons=100):
        # dane wejściowe
        self.inputs = inputs
        # liczba neuronow
        self.nNeurons = nNeurons
        # wspolczynnik uczenia
        self.learningRate0 = learningRateMax
        self.learningRate = 0
        # promien sasiedztwa
        self.radius = 0
        # potencjaly neuronow
        self.potentials = np.zeros(self.nNeurons)
        self.potentials[:] = 0.75 + 1 / (self.nNeurons - 1)
        # wagi zainicjalizowane losowo - kolumna oznacza jeden neuron
        self.weights = np.zeros((self.inputs.shape[1], self.nNeurons))
        for i in range(self.nNeurons):
            for j in range(self.inputs.shape[1]):
                self.weights[j][i] = random.uniform(-12, 12)
        # tablice do przechowywania poczatkowego polozenia neuronow
        self.NX0 = []
        self.NY0 = []
        # tablice do przechowywania koncowego polozenia neuronow
        self.NX = []
        self.NY = []
        self.error = 0

        for i in range(self.nNeurons):
            for j in range(self.inputs.shape[1]):
                self.NX0.append(self.weights[0][i])
                self.NY0.append(self.weights[1][i])

    def train(self):
        random.shuffle(self.inputs)
        self.error = 0
        for t in range(self.inputs.shape[0]):
            self.update(t)
        self.error /= self.inputs.shape[0]
        for i in range(self.nNeurons):
            self.NX.append(self.weights[0][i])
            self.NY.append(self.weights[1][i])
        plot.title('Chart for ' + self.nNeurons.__str__() + ' neurons : error = ' + self.error.__str__()
                   + '\n' + 'Parameters: learning rate( ' + learningRateMax.__str__() + ' - ' + learningRateMin.__str__() + ' )'
                   + ', radius( ' + radiusMax.__str__() + ' - ' + radiusMin.__str__() + ' )')
        plot.plot(self.inputs[:, 0], self.inputs[:, 1], 'ro', label='testing points')
        plot.xlabel('x')
        plot.ylabel('y')
        plot.plot(self.NX0, self.NY0, 'yo', label='initial neurons')
        plot.plot(self.NX, self.NY, 'bo', label='final neurons')
        plot.xlim(-15, 15)
        plot.ylim(-15, 15)
        plot.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=3)
        plot.show()

    def update(self, t):
        # obliczenie odleglosci euklidesowej pomiedzy neuronami a inputem
        input = self.inputs[t]
        distances = np.zeros(self.nNeurons)
        for i in range(self.nNeurons):
            for j in range(self.inputs.shape[1]):
                distances[i] += (input[j] - self.weights[j][i]) * (input[j] - self.weights[j][i])
            distances[i] = math.sqrt(distances[i])

        # zaktualizowanie learning rate
        self.learningRate = self.learningRate0 * math.pow(learningRateMin / self.learningRate0,
                                                          (t / self.inputs.shape[0]))
        # zaktualizowanie promienia sasiedztwa
        self.radius = radiusMax * math.pow(radiusMin / radiusMax, (t / self.inputs.shape[0]))

        # wybranie BMU czyli neuronu ktory jest najblizszy wektorowi wejsciowemu
        for i in range(self.nNeurons):
            if self.potentials[i] > potentialMin:
                position = i
                break

        for i in range(self.nNeurons):
            if distances[i] < distances[position] and self.potentials[i] > potentialMin:
                position = i

        # obliczenie odleglosci neuronow od neuronu zwycieskiego
        distancesFromBMU = np.zeros(self.nNeurons)
        for i in range(self.nNeurons):
            for j in range(self.inputs.shape[1]):
                distancesFromBMU[i] += math.pow((self.weights[j][i] - self.weights[j][position]), 2)
            distancesFromBMU[i] = math.sqrt(distancesFromBMU[i])

        # zaktualizowanie wag neuronow z wykorzstaniem funkcji gaussa
        for i in range(self.nNeurons):
            if distancesFromBMU[i] < self.radius and self.potentials[i] > potentialMin:
                for j in range(self.inputs.shape[1]):
                    self.weights[j][i] += self.learningRate * math.exp(
                        -(math.pow(distancesFromBMU[i], 2)) / (2 * math.pow(self.radius, 2))) * (
                                                  input[j] - self.weights[j][i])

        # aktualizacja potencjalow
        for i in range(self.nNeurons):
            if i == position:
                self.potentials[i] -= potentialMin
            else:
                self.potentials[i] += 1 / self.nNeurons

        # obliczenie bledu kwantyzacji w t-ej iteracji
        self.error += math.pow((distances[position]), 2) / self.inputs.shape[1]

    def animate(self, i):
        self.NX0 = []
        self.NY0 = []
        radius = round(self.radius, 4)
        learningRate = round(self.learningRate, 4)
        text = 'Iteration: ' + str(i) + '\n' + 'Parameters: learning rate: ' + learningRate.__str__() \
               + ', radius: ' + radius.__str__()
        self.update(i)
        for i in range(self.nNeurons):
            self.NX0.append(self.weights[0][i])
            self.NY0.append(self.weights[1][i])
        ax1.clear()
        ax1.plot(self.inputs[:, 0], self.inputs[:, 1], 'ro', label='training points')
        ax1.set_xlim([-15, 15])
        ax1.set_ylim([-15, 15])
        ax1.title.set_text(text)
        ax1.plot(self.NX0, self.NY0, 'bo', label='neurons position')
        ax1.legend(loc="upper right")


def randPoints(x1=-5, x2=5, numberOfPoints=10000):  # mozliwosc wylosowania punktow dla sinusoidy
    points = np.zeros((numberOfPoints, 2))
    for i in range(numberOfPoints):
        points[i][0] = random.uniform(x1, x2)
        points[i][1] = math.sin(points[i][0])
    return points


# Odczytanie pliku
data = np.array(pd.read_csv('daneTestowe.txt', names=['x', 'y'], header=None))

som1 = Som(data)
som1.train()

############ dla wylosowanej sinusoidy
# points = randPoints()
# som2 = Som(points)
# som2.train()

############ animacja
# som3 = Som(data)
# ani = animation.FuncAnimation(fig, som3.animate, frames=10000, interval=500)
# plot.show()


