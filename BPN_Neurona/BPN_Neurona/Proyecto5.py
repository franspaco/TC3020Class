import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt

def readFile(fileName):
    xIn = []
    yIn = []
    with open(fileName) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            xIn.append([])
            xIn[-1].append(float(1))
            xIn[-1].append(float(row[0]))
            xIn[-1].append(float(row[1]))
            yIn.append(int(row[2]))

    return np.matrix(xIn).T, np.matrix(yIn)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

vsig = np.vectorize(sigmoid)

def dif(x):
    if x >= 0.5:
        return 1
    else:
        return 0

vdif = np.vectorize(dif)

def funcionCostoSig(W, X, Y):
    x = X.T
    y = Y.T
    m = x.shape[0]
    J = 0
    hip = vsig(x*W)
    J = -y.T * np.log(hip) - (1-y).T * np.log(1-hip)
    J /= m
    return J

def funcionCostoLin(W, X, y):
    m = X.shape[1]
    Z = W.T * X
    J = np.square(Z-y).sum()
    J /= m
    return J

def bpnUnaNeuronaSigmoidal (nn_params, input_layer_size, X, y, alpha, activacion):
    W = nn_params
    m = X.shape[1]
    sigm = activacion == "sigmoidal"
    costs = []
    running = True
    count = 0
    while running:
        running = False
        Z = W.T * X
        if sigm:
            cost = funcionCostoSig(W, X, y).sum()
            A = vsig(Z)
        else:
            cost = funcionCostoLin(W, X, y).sum()
            A = Z
        costs.append(cost)
        dz = A - Y
        dw = (1/m) * X * (dz.T)
        W += -alpha * dw
        #print(str(cost) + "   " + str(W.T), end='\r')
        if (len(costs) < 100) or deltaIsBig(cost, costs[-2]):
            running = True
        if count > 10000:
            running = False
        count += 1
    return W, costs

def deltaIsBig(current, last):
    return abs(current-last) > 0.00001

def sigmoidGradiente(Z, Y):
    A = vsig(Z)
    

def linealGradiante(z):
    pass

def randInicializaPesos(L_in):
    w = []
    w.append(1)
    for num in range(1, L_in):
        w.append(random.uniform(-0.12, 0.12))
    return np.matrix(w).T

def prediceRNYaEntrenada(X, nn_params, activacion):
    Z = nn_params.T * X
    if activacion == "sigmoidal":
        A = vsig(Z)
        return vdif(A)
    else:
        return Z

def prediceSig(list, W):
    list = [float(i) for i in list]
    l = [1.0] + list
    l = np.matrix(l).T
    a = prediceRNYaEntrenada(l, W, "sigmoidal")
    return a.sum()

def prediceLin(list, W, med, ran):
    list = [float(i) for i in list]
    l = [1.0] + list
    l = np.matrix(l)
    l = normalizar(l, med, ran)
    a = prediceRNYaEntrenada(l.T, W, "lin")
    return a.sum()

def graficaDatos(X, Y, theta):
    plt.clf()
    m = X.shape[0]
    for n in range(0,m):
        if bool(Y[n,0]):
            plt.plot(X[n,1], X[n,2], 'rx')
        else:
            plt.plot(X[n,1], X[n,2], 'bo')
    mi = min(X[:,1]).sum()
    ma = max(X[:,1]).sum()
    rr = np.arange(mi, ma,(ma-mi)/1000)
    plt.plot(rr, f(rr, theta), 'g')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def f(x, theta):
    return -theta[0]/theta[2]-(theta[1]/theta[2])*(x)

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


def normalizar(X, med, ran):
    m = X.shape[0]
    n = X.shape[1]
    for x in range(1, n):
        for y in range(0, m):
            X[y,x] -= med[x-1]
            X[y,x] /= ran[x-1]
    return X


if __name__ == '__main__':

    ## HOUSES
    print("\n\nAprendiendo houses")
    X, Y = readFile('houses.txt')
    Xt, med, ran = normalizacionDeCaracteristicas(X.T)
    X = Xt.T
    nn_params = randInicializaPesos(3)
    W, costsHouses = bpnUnaNeuronaSigmoidal(nn_params, 3, X, Y, 0.03, "lineal")

    print("Pesos:")
    print(W)

    params = [1985,4]
    res = 299900
    print("Prediccion para: " + str(params) + " (Esperado: " + str(res) + ")")
    print(prediceLin(params, W, med, ran))

    params = [2104,3]
    res = 399900
    print("Prediccion para: " + str(params) + " (Esperado: " + str(res) + ")")
    print(prediceLin(params, W, med, ran))

    plt.plot(costsHouses, 'r')
    plt.xlabel('Iteracion')
    plt.ylabel('Costo')
    plt.show()
    

    ## EXAMS
    X, Y = readFile('exams.txt')
    nn_params = randInicializaPesos(3)
    print("\n\nAprendiendo Exams")
    X[1,:] /= 100
    X[2,:] /= 100
    W, costsEx = bpnUnaNeuronaSigmoidal(nn_params, 1, X, Y, 3, "sigmoidal")
    print("Pesos:")
    print(W)
    graficaDatos(X.T, Y.T, W.A1)

    plt.plot(costsEx)
    plt.xlabel('Iteracion')
    plt.ylabel('Costo')
    plt.show()

    ## AND
    X, Y = readFile('and.txt')
    nn_params = randInicializaPesos(3)
    print("\n\nAprendiendo AND")
    W, costsAnd = bpnUnaNeuronaSigmoidal(nn_params, 3, X, Y, 3, "sigmoidal")
    print("Pesos:")
    print(W)
    print('\n a b | a and b')
    print(' 0 0 | ', end='')
    a = prediceSig([0,0], W)
    print(a)
    print(' 0 1 | ', end='')
    a = prediceSig([0,1], W)
    print(a)
    print(' 1 0 | ', end='')
    a = prediceSig([1,0], W)
    print(a)
    print(' 1 1 | ', end='')
    a = prediceSig([1,1], W)
    print(a)
    graficaDatos(X.T, Y.T, W.A1)

    ## OR
    X, Y = readFile('or.txt')
    nn_params = randInicializaPesos(3)
    print("\n\nAprendiendo OR")
    W, costsOr = bpnUnaNeuronaSigmoidal(nn_params, 3, X, Y, 3, "sigmoidal")
    print("Pesos:")
    print(W)
    print('\n a b | a or b')
    print(' 0 0 | ', end='')
    a = prediceSig([0,0], W)
    print(a)
    print(' 0 1 | ', end='')
    a = prediceSig([0,1], W)
    print(a)
    print(' 1 0 | ', end='')
    a = prediceSig([1,0], W)
    print(a)
    print(' 1 1 | ', end='')
    a = prediceSig([1,1], W)
    print(a)
    graficaDatos(X.T, Y.T, W.A1)

    plotAnd, = plt.plot(costsAnd, 'b', label='AND')
    plotOr, = plt.plot(costsOr, 'r', label='OR')
    plt.xlabel('Iteracion')
    plt.ylabel('Costo')
    plt.legend(handles=[plotAnd, plotOr])
    plt.show()

