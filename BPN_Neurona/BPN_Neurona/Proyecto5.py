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
    #plt.ion()
    while running:
        #print(W.A1)
        #graficaDatos(X.T, Y.T, W.A1)
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
        W += -alpha * dw
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

def prediceSig(list, W):
    l = [1] + list
    l = np.matrix(l)
    a = prediceRNYaEntrenada(l.T, W, "sigmoidal")
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
    #plt.pause(0.05)
    plt.show()

def f(x, theta):
    return -theta[0]/theta[2]-(theta[1]/theta[2])*(x)

if __name__ == '__main__':
    
    X, Y = readFile('exams.txt')
    nn_params = randInicializaPesos(3)
    print("Aprendiendo Exams")
    X[1,:] /= 100
    X[2,:] /= 100
    print(X.T)
    W, costsEx = bpnUnaNeuronaSigmoidal(nn_params, 1, X, Y, 3, "sigmoidal")
    print("Pesos:")
    print(W)
    graficaDatos(X.T, Y.T, W.A1)
    #print(prediceSig([34.62365962451697,78.0246928153624], W))

    a = prediceRNYaEntrenada(X, W, "sigmoidal")
    for num in range(X.shape[1]):
        print(str(a[0,num]) + " - " + str(Y[0, num]))


    plt.plot(costsEx, 'g')
    plt.show()

    exit()

    X, Y = readFile('and.txt')
    nn_params = randInicializaPesos(3)
    print("Aprendiendo AND")
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

    plt.plot(costsAnd, 'b')
    plt.plot(costsOr, 'r')
    plt.show()