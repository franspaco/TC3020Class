
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
            xIn[-1].append(float(row[0])/100)
            xIn[-1].append(float(row[1])/100)
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
    return sigmoidal(x * np.matrix(theta).T)

def funcionCosto(theta, X, Y):
    m = X.shape[0]
    J = 0
   # for i in range(0,m):
       # y = Y[i,0]
        #x = X[i]
       # hip = h(x, theta)
        #J += -y*np.log(hip)-((1-y)*np.log(1-hip))

    #J /= m

    grad = X.T * (h(X, theta) - Y)
    grad /= m

    return J, grad

def aprende(theta, X, Y, iteraciones):
    error = []
    for num in range(0, iteraciones):
        j, grad = funcionCosto(theta, X, Y)
        #error.append(j.sum())
        theta -= 0.1 * grad.A1
        #print(str(num) + " " + str(j) + " " + str(theta))
    return theta, error

def predice(theta, X):
    X = np.matrix(X) / 100
    p = h(X,theta)
    if p >= 0.5:
        return 1
    else:
        return 0

if __name__ == '__main__':
    fileName = 'ex2data1.txt'
    X, Y = readFile(fileName)
    #graficaDatos(X,Y, 1)
    theta, error = aprende([0,0], X, Y, 1500)
    print(theta)
    print(h([45, 85], theta))
    predice(theta, [45, 85])
    plt.plot(error)
    plt.show()