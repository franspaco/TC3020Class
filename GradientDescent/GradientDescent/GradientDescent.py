import csv
import numpy as np
import matplotlib.pyplot as plt

    #answer: [[-3.89578088],
    #         [ 1.19303364]]

def readFile(fileName):
    xIn = []
    yIn = []
    print('Reading ' + fileName)
    with open(fileName) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            xIn.append(float(row[0]))
            yIn.append(float(row[1]))

    print('Reading OK!')
    return xIn, yIn

def calculaCosto(X, Y, theta):
    h = X * (np.matrix(theta).T)
    cost = h - Y
    return cost

def gradenteDescendente(x, y, theta, alpha, iteraciones):
    m = len(x)
    print(m)
    a = np.matrix( np.ones_like(x))
    b = np.matrix(x)
    X = np.concatenate((a,b)).T
    Y = np.matrix(y).T
    for num in range(iteraciones):
        cost = calculaCosto(X, Y, theta)
        loss1 = b * cost 
        avgError0 = cost.sum() / m
        avgError1 = loss1.sum() / m
        temp0 = theta[0] - alpha * avgError0
        temp1 = theta[1] - alpha * avgError1
        theta[0] = temp0
        theta[1] = temp1
    print(theta)
    graficaDatos(x, y, theta)
    return;

def graficaDatos(x, y, theta):
    plt.plot(x, y, 'ro')
    rr = np.arange(min(x), max(x), 0.1)
    plt.plot(rr, h(theta, rr))
    plt.title('Fitting with y=' + str(theta[0]) + ' + ' + str(theta[1]) + 'x')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.show()
    return;

def h(theta, X):
    l = []
    for i in X:
        l.append(theta[0] + theta[1]*i)
    return l

if __name__ == '__main__':
    fileName = 'ex1data1.txt'
    x, y = readFile(fileName)
    gradenteDescendente(x, y, [0, 0], 0.01, 10000)