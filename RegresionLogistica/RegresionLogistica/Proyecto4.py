
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

def readFile(fileName):
    xIn = []
    yIn = []
    print('Reading ' + fileName)
    with open(fileName) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            xIn.append([])
            #xIn[-1].append(float(1))
            xIn[-1].append(float(row[0]))
            xIn[-1].append(float(row[1]))
            yIn.append(int(row[2]))

    print('Reading OK!')
    xIn = np.matrix(xIn)
    yIn = np.matrix(yIn).T
    return xIn, yIn

def graficaDatos(X, Y, theta):
    m = X.shape[0]
    for n in range(0,m):
        if bool(Y[n,0]):
            plt.plot(X[n,0], X[n,1], 'rx')
        else:
            plt.plot(X[n,0], X[n,1], 'bo')
    plt.show()

def sigmoidal(z):
    return 1/(1+np.exp(-z))

def h(x, theta):
    return sigmoidal((x * np.matrix(theta).T).sum())

def funcionCosto(theta, X, Y):
    m = X.shape[0]
    J = 0
    for i in range(0,m):
        y = Y[i,0]
        x = X[i]
        h = h(x, theta)
        J += -y*np.log(h)-((1-y)*np.log(1-h))

    J /= m

    return J

def aprende(theta, X, y, iteraciones):


def predice(theta, X):
    p = h(X,theta)
    if p >= 0.5:
        return 1
    else:
        return 0

if __name__ == '__main__':
    fileName = 'ex2data1.txt'
    X, Y = readFile(fileName)
    #graficaDatos(X,Y, 1)
    rr=np.arange(-5, 5, 0.1)
    plt.plot(rr, sigmoidal(rr))
    plt.show()