
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

def h(x, theta):
    return x * np.matrix(theta).T

def funcionCosto(theta, X, Y):
    m = X.shape[0]
    J = 0

    hip = h(X, theta)
    J = np.power(hip - Y, 2)
    J = J.sum()
    J /= m * 2

    grad = X.T * (h(X, theta) - Y)
    grad /= m

    return J, grad

def gradenteDescendente(X, Y, theta, alpha, iteraciones):
    m = X.shape[0]
    print(m)
    b = np.matrix(X[:,1].A1)
    error = []
    for num in range(iteraciones):
        j, grad = funcionCosto(theta, X, Y)
        error.append(j.sum())
        #theta -= alpha * grad.A1 
        theta -= alpha * grad.A1 / (num + 1)
    print(theta)
    #graficaDatos(X, Y, theta)
    return theta, error

def graficaDatos(X, Y, theta):
    x = X[:,1].A1
    y = Y.A1
    plt.plot(x, y, 'ro')
    rr = np.arange(min(x), max(x), 0.1)
    plt.plot(rr, ht(theta, rr))
    plt.title('Fitting with y=' + str(theta[0]) + ' + ' + str(theta[1]) + 'x')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.show()
    return;

def ht(theta, X):
    l = []
    for i in X:
        l.append(theta[0] + theta[1]*i)
    return l

if __name__ == '__main__':
    fileName = 'ex1data1.txt'
    X, Y = readFile(fileName)
    theta, error = gradenteDescendente(X, Y, [0, 0], 0.01, 1500)
    print(error[-1])
    plt.plot(error)
    plt.ylabel('Costo')
    plt.xlabel('Iteracion')
    plt.show()
    graficaDatos(X, Y, theta)

# Alpha cambia
# costo = 5.87546776961

# Alpha constante
# costo = 4.48341145337