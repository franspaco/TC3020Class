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
            xIn.append([])
            xIn[-1].append(float(1))
            xIn[-1].append(float(row[0]))
            yIn.append(float(row[1]))

    print('Reading OK!')
    xIn = np.matrix(xIn)
    yIn = np.matrix(yIn).T
    return xIn, yIn

def calculaCosto(X, Y, theta):
    h = X * (np.matrix(theta).T)
    cost = h - Y
    return cost

def gradenteDescendente(X, Y, theta, alpha, iteraciones):
    m = X.shape[0]
    print(m)
    b = np.matrix(X[:,1].A1)
    for num in range(iteraciones):
        cost = calculaCosto(X, Y, theta)
        loss1 = X[:,1].T * cost
        avgError0 = cost.sum() / m
        avgError1 = loss1.sum() / m
        temp0 = theta[0] - alpha * avgError0
        temp1 = theta[1] - alpha * avgError1
        theta[0] = temp0
        theta[1] = temp1
        #print(avgError0)
    print(theta)
    #graficaDatos(X, Y, theta)
    return theta;

def graficaDatos(X, Y, theta):
    x = X[:,1].A1
    y = Y.A1
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
    X, Y = readFile(fileName)
    theta = gradenteDescendente(X, Y, [0, 0], 0.01, 1500)
    graficaDatos(X, Y, theta)