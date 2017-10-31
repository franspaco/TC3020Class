
import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt


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

def funCosto(h, y):
    m = h.shape[1]

    J = - np.multiply(y, np.log(h)) - np.multiply(1-y, np.log(1-h))
    J = J.sum()
    J /= m

    return J

def entrenaRNB(input_layer_size, hidden_layer_size, num_labels, X, y):
    alpha = 1
    m = X.shape[1]
    # A0 is 400xm
    A0 = X
    # W1 is 25x400
    W1 = randInicializacionPesos(input_layer_size, hidden_layer_size)
    # W2 is 10x25
    W2 = randInicializacionPesos(hidden_layer_size, num_labels)

    b1 = np.ones((hidden_layer_size, 1));
    b2 = np.ones((num_labels, 1));

    costs = []
    running = True
    count = 0
    while running:
        running = False
        count += 1
        # Hidden layer
        # Z1 is 25xm
        Z1 = W1 * A0 + b1
        # A1 is 25xm
        A1 = vsig(Z1)

        # Output layer
        # Z2 is 10xm
        Z2 = W2 * A1 + b2
        # A2 is 10xm
        A2 = vsig(Z2)

        J = funCosto(A2, y)
        print('IT #' + str(count) + ' J=' + str(J), end='\r')
        costs.append(J)

        #Back
        # 10xm = (10xm - 10xm)
        dz2 = A2 - y
        # 10x25 = 10xm * mx25
        dw2 = (1/m) * dz2 * A1.T
        db2 = (1/m) * np.sum(np.array(dz2), axis=1, keepdims=True)

        # 25xm = (25x10 * 10xm) .* 25xm
        dz1 = np.multiply(W2.T * dz2, sigmoidalGradienteA(A1))
        # 25x400 = 25xm * mx400
        dw1 = (1/m) * dz1 * A0.T
        db1 = (1/m) * np.sum(np.array(dz1), axis=1, keepdims=True)

        W2 += -alpha * dw2
        b2 += -alpha * db2
        W1 += -alpha * dw1
        b1 += -alpha * db1

        if (len(costs) < 100) or deltaIsBig(costs[-1], costs[-2]):
            running = True

        if count > 2000:
            running = False
            print('\nSTOP by Iterations')

    return W1, b1, W2, b2, costs




def entrenaRN(input_layer_size, hidden_layer_size, num_labels, X, y):
    alpha = 0.01
    #alpha = 0.0003
    #alpha = 0.00001

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
        # A1 is 25xm
        A1 = vsig(Z1)
        # A1 is 26xm
        A1 = np.concatenate((np.ones((1,A1.shape[1])),A1), axis=0)
        # A1 is 26xm

        # Output layer

        # Z2 is 10xm
        Z2 = W2 * A1
        # A2 is 10xm
        A2 = vsig(Z2)

        # COST 
        J = funCosto(A2, y)
        print('IT #' + str(count) + ' J=' + str(J), end='\r')
        costs.append(J)

        # BACK PROPAGATION

        # 10xm = (10xm - 10xm)
        dz2 = A2 - y

        # 26xm = (26x10 * 10xm) .* 26xm
        dz1 = np.multiply(W2.T * dz2, sigmoidalGradienteA(A1))

        # 25xm
        dz1 = dz1[1:]

        # PARAM UPDATE
        dw2 = (1/m) * dz2 * A1.T
        dw1 = (1/m) * dz1 * A0.T
        W2 += -alpha * dw2
        W1 += -alpha * dw1

        if (len(costs) < 100) or deltaIsBig(costs[-1], costs[-2]):
            running = True
        else:
            print('\nSTOP by Delta')
            pass
        if( len(costs) > 100 and costs[-1] > costs[-2]):
            #running = False
            #print('\nSTOP by Bounce')
            pass
        if count > 2000:
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

def prediceRNYaEntrenadaB(X, W1, b1, W2, b2):
    # A0 is 401xm
    A0 = X
    Z1 = W1 * A0 + b1
    # A1 is 25xm
    A1 = vsig(Z1)

    # Output layer
    # Z2 is 10xm
    Z2 = W2 * A1 + b2
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
    X, y = readFile('digitos2500.txt')
    print('\nSHAPES:')
    print(X.shape)
    print(y.shape)
    #return
    print('\nTRAINING:')
    W1, b1, W2, b2, costs = entrenaRNB(400, 25, 10, X, y)
    #print(W1)
    #print(W2)
    string = '_500_25'
    np.save('w1' + string + '.npy', W1)
    np.save('w2' + string + '.npy', W2)
    np.save('b1' + string + '.npy', b1)
    np.save('b2' + string + '.npy', b2)

    plt.plot(costs)
    plt.show()
    return W1, b1, W2, b2

def sacaCosto(X, y, W1, W2):
    A0 = np.concatenate((np.ones((1,X.shape[1])),X), axis=0)
    # Hidden layer
    # Z1 is 25xm
    Z1 = W1 * A0
    # A1 is 25xm
    A1 = vsig(Z1)
    # A1 is 26xm
    A1 = np.concatenate((np.ones((1,A1.shape[1])),A1), axis=0)
    # A1 is 26xm

    # Output layer
    # Z2 is 10xm
    Z2 = W2 * A1
    # A2 is 10xm
    A2 = vsig(Z2)
    J = funCosto(A2, y)
    print('COSTO: ' + str(J))

def predictions(W1, b1, W2, b2):
    Xt, yt = readFile('digitos.txt')
    total = Xt.shape[1]
    sucess = 0
    for i in range(Xt.shape[1]):
        val = prediceRNYaEntrenadaB(Xt[:,i],W1, b1, W2, b2)
        exp = vetorToClass(yt[:,i])
        print("\nEXPECTED: " + str(exp))
        print("FOUND:    " + str(val))
        if val == exp:
            sucess += 1
        #drawImg(Xt[:,i], exp)

    print("Correct: " + str(sucess) + '/' + str(total) + ' => ' + str(sucess/total*100) + '%')

def main():
    W1, b1, W2, b2 = training()

    #W1 = np.load('w1_500_20.npy')
    #W2 = np.load('w2_500_20.npy')

    predictions(W1, b1, W2, b2)



if __name__ == '__main__':
    main()