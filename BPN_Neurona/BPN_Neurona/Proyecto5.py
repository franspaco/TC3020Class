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

def funcionCosto(W, X, Y):
    x = X.T
    y = Y.T
    m = x.shape[0]
    J = 0
    hip = vsig(x*W)
    J = -y.T * np.log(hip) - (1-y).T * np.log(1-hip)
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
        cost = funcionCosto(W, X, y).sum()
        costs.append(cost)
        Z = W.T * X
        if sigm:
            A = vsig(Z)
        else:
            A = Z
        dz = A - Y
        dw = (1/m) * X * (dz.T)
        #db = (1/m) * np.sum(dz)
        W += -alpha * dw
        if (len(costs) == 1) or deltaIsBig(cost, costs[-2]):
            running = True
        if count > 10000:
            running = False
        count += 1
    return W, costs

def deltaIsBig(current, last):
    return abs(current-last) > 0.000001

def sigmoidGradiente(z):
    pass

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

def prediceSig(list, W):
    l = [1] + list
    l = np.matrix(l)
    a = prediceRNYaEntrenada(l.T, W, "sigmoidal")
    return a.sum()

if __name__ == '__main__':
    #and - 3
    #or  - 3
    X, Y = readFile('or.txt')
    print(X)
    print(Y)
    nn_params = randInicializaPesos(3)
    W, costs = bpnUnaNeuronaSigmoidal(nn_params, 3, X, Y, 3, "sigmoidal")

    print(W)
    a = prediceSig([0,0], W)
    print(a)
    a = prediceSig([0,1], W)
    print(a)
    a = prediceSig([1,0], W)
    print(a)
    a = prediceSig([1,1], W)
    print(a)

    plt.plot(costs)
    plt.show()