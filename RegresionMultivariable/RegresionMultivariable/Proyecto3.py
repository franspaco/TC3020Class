
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
            xIn.append([])
            xIn[-1].append(float(1))
            xIn[-1].append(float(row[0]))
            xIn[-1].append(float(row[1]))
            yIn.append(float(row[2]))

    print('Reading OK!')
    xIn = np.matrix(xIn)
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
        cost_hist.append(calculaCosto(X, Y, theta))
        derivadaCost = calculaCostoDer(X, Y, theta)
        for indx in range(0, n):
            err = X[:,indx].T * derivadaCost
            avgErr = err.sum() / m
            theta[indx] -= alpha * avgErr
    return theta, cost_hist;

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
    X = np.append([1], X)
    theta = np.matrix(theta).T
    return (X*theta).sum()

def graficaError(J_Historial):
    plt.plot(J_Historial)
    plt.ylabel('Costo')
    plt.xlabel('Iteraciones')
    plt.show()

def mejorAlpha(X, Y, theta0, alpha, iterations):
    print("\nRound: 1, alpha: " + str(alpha))
    theta, cost = gadienteDescendenteMultivariable(X, Y, theta0, alpha, iterations)
    graficaError(cost)
    print(cost[-1])
    print(theta)

    bestalpha = alpha
    minCost = cost[-1]
    bestTheta = theta
    for num in range(1,10):
        alpha *= 3
        print("Round: " + str(num+1) + ", alpha: " + str(alpha))
        theta, cost = gadienteDescendenteMultivariable(X, Y, theta0, alpha, iterations)
        if math.isnan(cost[-1]):
            print("Diverge")
            continue
        graficaError(cost)
        if cost[-1] < minCost:
            minCost = cost[-1]
            bestalpha = alpha
        print(cost[-1])
        print(theta)
    print("Best Alpha:")
    print(bestalpha)
    print("Minimum Cost:")
    print(minCost)

if __name__ == '__main__':
    fileName = 'ex1data2.txt'
    X, Y = readFile(fileName)
    theta0 = [0, 0, 0]

    alpha = 0.729
    iterations = 1500
    X, mu, sigma = normalizacionDeCaracteristicas(X)

    thetaFin = ecuacionNormal(X, Y)
    costoFin = calculaCosto(X, Y, thetaFin.A1)
    print("Ecuación normal:")
    print("Best theta:")
    print(thetaFin)
    print("BestCost:")
    print(costoFin)

    theta, cost = gadienteDescendenteMultivariable(X, Y, theta0, alpha, 1500)
    print("\nGradiente descendente:")
    print(theta)
    graficaError(cost)
    #graficaDatos(X, Y, theta)

# Sin normalizar
#[[ 89597.9095428 ]
# [   139.21067402]
# [ -8738.01911233]]

# Normalizado
#[[ 340412.65957447]
# [ 504777.90398791]
# [ -34952.07644931]]