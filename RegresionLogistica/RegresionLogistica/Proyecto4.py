
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
            xIn[-1].append(float(1))
            xIn[-1].append(float(row[0])/100)
            xIn[-1].append(float(row[1])/100)
            yIn.append(int(row[2]))

    print('Reading OK!')
    xIn = np.matrix(xIn)
    yIn = np.matrix(yIn).T
    return xIn, yIn

def sigmoidal(z):
    return 1/(1+np.exp(-z))

def h(x, theta):
    return sigmoidal(x * np.matrix(theta).T)

def funcionCosto(theta, X, Y):
    m = X.shape[0]
    J = 0

    hip = h(X, theta)
    for i in range(0,m):
        y = Y[i,0]
        J += -y*np.log(hip[i,0])-((1-y)*np.log(1-hip[i,0]))

    J /= m

    grad = X.T * (h(X, theta) - Y)
    grad /= m

    return J, grad

def aprende(theta, X, Y, iteraciones):
    error = []
    for num in range(0, iteraciones):
        j, grad = funcionCosto(theta, X, Y)
        error.append(j.sum())
        theta -= 3 * grad.A1
        print(str(num) + " " + str(j) + " " + str(theta))
    return theta, error

def predice(theta, X):
    X = np.matrix(X) / 100
    X = np.matrix(np.concatenate(([1], X.A1)))
    p = h(X,theta)
    if p >= 0.5:
        return 1
    else:
        return 0

def graficaDatos(X, Y, theta):
    m = X.shape[0]
    for n in range(0,m):
        if bool(Y[n,0]):
            plt.plot(X[n,1]*100, X[n,2]*100, 'rx')
        else:
            plt.plot(X[n,1]*100, X[n,2]*100, 'bo')
    rr = np.arange(20,100,1)
    plt.plot(rr, f(rr, theta)*100, 'g')
    plt.xlabel('Examen 1')
    plt.ylabel('Examen 2')
    plt.show()

def f(x, theta):
    return -theta[0]/theta[2]-(theta[1]/theta[2])*(x/100)
    #return 1.25-1.1971*x

if __name__ == '__main__':
    fileName = 'ex2data1.txt'
    X, Y = readFile(fileName)
    theta, error = aprende([0,0,0], X, Y, 1500)
    print(theta)
    plt.plot(error)
    plt.ylabel('Costo')
    plt.xlabel('Iteracion')
    plt.show()
    graficaDatos(X,Y, theta)
    #print(h([1, 0.45, 0.85], theta))
    print(predice(theta, [45, 85]))