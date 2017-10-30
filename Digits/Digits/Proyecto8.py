
import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import minimize


def sig(z):
    if z < 0:
        return 1 - 1/(1 + math.exp(z))
    else:
        return 1/(1 + math.exp(-z))
    #return 1 / (1 + math.exp(-z))
    #try:
        #return 1 / (1 + math.exp(-z))
    #except:
        #return 0

vsig = np.vectorize(sig)

def classToVector(val):
    val = int(val) % 10
    temp = [0,0,0,0,0,0,0,0,0,0]
    temp[val] = 1
    return temp

def vetorToClass(vec):
    for i, v in enumerate(vec):
        if v == 1:
            return i

def readFile(fileName):
    xIn = []
    yIn = []
    with open(fileName) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        for row in spamreader:
            if len(row) < 400:
                continue
            xIn.append([])
            for indx, val in enumerate(row):
                if indx == 0 or val == '' or val == ' ':
                    continue
                if indx == 401:
                    yIn.append(classToVector(val))
                    break
                num = float(val)
                if num > 1:
                    num = 1
                elif num < -1:
                    num = -1
                xIn[-1].append(num)      
    return np.matrix(xIn).T, np.matrix(yIn).T

def ps(x):
    for sub in x:
        print(len(sub), end=" ")
    print("")

def funCosto(h, y, W1, W2, lam):
    m = h.shape[1]

    J = - np.multiply(y, np.log(h)) - np.multiply(1-y, np.log(1-h))
    J = J.sum()
    
    J /= m

    reg = (lam /(2.0*m)) * (np.power(W1[:,1:], 2).sum() + np.power(W2[:,1:], 2).sum())

    return J + reg

def entrenaRN(input_layer_size, hidden_layer_size, num_labels, X, y):
    alpha = 0.1
    #alpha = 0.0003
    #alpha = 0.00001

    lam = 1

    m = X.shape[1]

    # X is 401xm
    X = np.concatenate((np.ones((1,X.shape[1])),X), axis=0)
    A0 = X
    # W1 is 25x401
    W1 = randInicializacionPesos(input_layer_size + 1, hidden_layer_size)
    # W2 is 10x26
    W2 = randInicializacionPesos(hidden_layer_size + 1, num_labels)

    #print('X:  ' + str(X.shape))
    #print('y:  ' + str(y.shape))
    #print('W1: ' + str(W1.shape))
    #print('W2: ' + str(W2.shape))

    costs = []
    running = True
    count = 0
    while running:
        running = False
        count += 1
        # Hidden layer

        # Z1 is 25xm
        Z1 = W1 * A0
        # tempA is 25xm
        A1 = vsig(Z1)
        # tempA is 26xm
        A1 = np.concatenate((np.ones((1,A1.shape[1])),A1), axis=0)
        # A1 is 26xm

        # Output layer

        # Z2 is 10xm
        Z2 = W2 * A1
        # A2 is 10xm
        A2 = vsig(Z2)

        # COST 
        J = funCosto(A2, y, W1, W2, 1)
        print('IT #' + str(count) + ' J=' + str(J), end='\r')
        costs.append(J)
        #print('J=  ' + str(J))

        # BACK PROPAGATION

        # ESTO EN UNA COPIA LITERAL DE CADA COSA.. maybe works?
        delta1 = np.zeros(W1.shape) 
        delta2 = np.zeros(W2.shape)

        for n in range(m):
            d_2 = y[:,n] - A2[:,n]
            d_1 = np.multiply(W2.T * d_2, sigmoidalGradienteA(A1[:,n]))
            d_1 = d_1[1:]
            delta2 += d_2 * A1[:,n].T
            delta1 += d_1 * A0[:,n].T

        W2grad =  alpha * (1/m) * delta2 + (lam/m) * np.concatenate((np.zeros((W2.shape[0],1)),W2[:,1:]), axis=1)
        W1grad =  alpha * (1/m) * delta1 + (lam/m) * np.concatenate((np.zeros((W1.shape[0],1)),W1[:,1:]), axis=1)

        W2 += W2grad
        W1 += W1grad

        # ESTO ES LO QUE ESTABA USANDO
        # 10xm = (10xm - 10xm) .* 10xm
        #d2 = np.multiply(A2 - y, sigmoidalGradienteA(A[2]))
        #d2 = A2 - y
        # NO SE CUAL DE ESTAS DOS USAR

        # 26xm = (26x10 * 10xm) .* 26xm
        #d1 = np.multiply(W2.T * d2, sigmoidalGradienteA(A1))

        # 25xm
        #d1 = d1[1:]

        # PARAM UPDATE
        #W2 -= alpha * (1/m) * d2 * A1.T + (lam/m) * np.concatenate((np.zeros((W2.shape[0],1)),W2[:,1:]), axis=1)
        #W1 -= alpha * (1/m) * d1 * A0.T + (lam/m) * np.concatenate((np.zeros((W1.shape[0],1)),W1[:,1:]), axis=1)

        if (len(costs) < 100) or deltaIsBig(costs[-1], costs[-2]):
            running = True
        else:
            print('\nSTOP by Delta')
            pass
        if( len(costs) > 100 and costs[-1] > costs[-2]):
            #running = False
            #print('\nSTOP by Bounce')
            pass
        if count > 200:
            running = False
            print('\nSTOP by Iterations')

    print('\nTRAINING DONE')
    return W1, W2, costs

def deltaIsBig(current, last):
    return abs(current-last) > 0.00000001


def sigmoidalGradiente(z):
    A = vsig(z)
    return np.multiply(A, 1-A)

def sigmoidalGradienteA(A):
    return np.multiply(A, 1-A)

def randInicializacionPesos(L_in, L_out):
    W = []
    for neuron in range(L_out):
        W.append([])
        for feature in range(L_in):
            W[-1].append(random.uniform(-0.12, 0.12))
    return np.matrix(W)

def prediceRNYaEntrenada(X,W1,W2):
    X = np.concatenate((np.ones((1,X.shape[1])),X), axis=0)
    Z1 = W1 * X
    # A1 is 25xm
    A1 = vsig(Z1)
    # A1 is 26xm
    A1 = np.concatenate((np.ones((1,A1.shape[1])),A1), axis=0)

    # Output layer

    # Z2 is 10xm
    Z2 = W2 * A1
    # A2 is 10xm
    A2 = vsig(Z2)
    max = np.argmax(A2)
    return max

def drawImg(data, val = '?'):
    plt.imshow(np.reshape(data, (20,20)),vmin=-1, vmax=1)
    plt.title('Drawing: ' + str(val))
    plt.show();

def training():
    print('\nREADING:')
    X, y = readFile('digitos50.txt')
    print('\nSHAPES:')
    print(X.shape)
    print(y.shape)
    #return
    print('\nTRAINING:')
    W1, W2, costs = entrenaRN(400, 25, 10, X, y)
    #print(W1)
    #print(W2)
    np.save('w1_5000_1.npy', W1)
    np.save('w2_5000_1.npy', W2)

    plt.plot(costs)
    plt.show()
    return W1, W2

def predictions(W1, W2):
    Xt, yt = readFile('digitos50.txt')
    val = prediceRNYaEntrenada(Xt,W1,W2)
    total = Xt.shape[1]
    sucess = 0
    for i in range(Xt.shape[1]):
        val = prediceRNYaEntrenada(Xt[:,i],W1,W2)
        exp = vetorToClass(yt[:,i])
        print("\nEXPECTED: " + str(exp))
        print("FOUND:    " + str(val))
        if val == exp:
            sucess += 1
        #drawImg(Xt[:,i], exp)

    print("Correct: " + str(sucess) + '/' + str(total) + ' => ' + str(sucess/total*100) + '%')

def main():
    W1, W2 = training()

    #W1 = np.load('w1_5000_1.npy')
    #W2 = np.load('w2_5000_1.npy')

    predictions(W1, W2)



if __name__ == '__main__':
    main()