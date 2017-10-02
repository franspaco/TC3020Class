
import csv
import numpy as np
import matplotlib.pyplot as plt
import math

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
    xIn = np.matrix(xIn).T
    yIn = np.matrix(yIn).T
    return xIn, yIn

def calculaCostoDer(X, Y, theta):
    h = X * (np.matrix(theta).T)
    cost = h - Y
    return cost

def calculaCosto(X, Y, theta):
    h = X * (np.matrix(theta).T)
    cost = h - Y
    cost = np.multiply(cost, cost)
    cost = cost.sum()
    cost /= (2 * (X.shape[0]))
    return cost

def gadienteDescendenteMultivariable(X, Y, theta, alpha, iteraciones):
    m = X.shape[0]
    n = X.shape[1]
    cost_hist = []
    for num in range(iteraciones):
        #cost_hist.append(calculaCosto(X, Y, theta))
        derivadaCost = calculaCostoDer(X, Y, theta)
        #print(derivadaCost)
        for indx in range(0, n):
            err = X[:,indx].T * derivadaCost
            avgErr = err.sum() / m
            print(str(alpha) + " * " + str(avgErr))
            theta[indx] -= alpha * avgErr
    return theta, cost_hist;

def polinomial(x, grad):
    if grad > x.shape[0]:
        print("Invalid input")
        return

    x = x.T
    X = np.matrix(np.ones_like(x));

    for num in range(1, (grad + 1)):
        l = np.power(x, num)
        X = np.concatenate((X, l))

    return X.T

def normalizacionDeCaracteristicas(X):
    m = X.shape[0]
    n = X.shape[1]
    ran = []
    med = []

    for i in range(1, n):
        ran.append(X[:,i].max() - X[:,i].min())
        med.append(X[:,i].sum() / m)

    for x in range(1, n):
        for y in range(0, m):
            X[y,x] -= med[x-1]
            X[y,x] /= ran[x-1]

    return X, med, ran

def ecuacionNormal(X,Y):
    return (((X.T) * X).I) * (X.T) * Y

def predicePrecio(X,theta):
    return X * theta

def graficaError(J_Historial):
    plt.plot(J_Historial)
    plt.ylabel('Costo')
    plt.xlabel('Iteraciones')
    plt.show()

def graficaDatos(X, Y, theta):
    x = X[:,1].A1
    y = Y.A1
    plt.plot(x, y, 'ro')
    rr = np.arange(min(x), max(x), 0.1)
    print(len(theta))
    vals = h(theta, rr)
    plt.plot(rr, vals)
    #plt.title('Fitting with y=' + str(theta[0]) + ' + ' + str(theta[1]) + 'x')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.show()
    return;

def h(theta, X):
    l = []
    for i in X:
        val = 0
        for num in range(len(theta)):
            val += theta[num] * (i**num)
        l.append(val)
    return l

if __name__ == '__main__':

    grad = 8

    X, Y = readFile('data.txt');
    X = polinomial(X, grad)

    tf = ecuacionNormal(X, Y)
    print(tf)

    graficaDatos(X, Y, tf.A1)

    theta0 = [0] * (grad+1)
    alpha = 0.0000001
    iteraciones = 1500
    #theta, costo = gadienteDescendenteMultivariable(X, Y, theta0, alpha, iteraciones)

    #print(theta)

