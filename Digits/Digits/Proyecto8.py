
import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def sig(z):
    try:
        return 1 / (1 + math.exp(-z))
    except:
        return 0

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
    #J = (-(np.log(h).T * y) - (np.log(1-h).T * (1-y))).sum()
    #J /= m
    #J = 0
    J = - np.multiply(y, np.log(h)) - np.multiply(1-y, np.log(1-h))
    J = J.sum()
    #for i in range(0,m):
        #for k in range(0,h.shape[0]):
            #J += -y[k, i] * np.log(h[k,i]) - (1-y[k, i])*np.log(1-h[k,i])
    J /= m
    return J

def entrenaRN(input_layer_size, hidden_layer_size, num_labels, X, y):
    alpha = 0.003
    #alpha = 0.0003
    #alpha = 0.00001
    m = X.shape[1]

    # X is 401xm
    X = np.concatenate((np.ones((1,X.shape[1])),X), axis=0)
    A = []
    A.append(X)
    # W1 is 25x401
    W1 = randInicializacionPesos(401, 25)
    # W2 is 10x26
    W2 = randInicializacionPesos(26, 10)

    print('X:  ' + str(X.shape))
    print('y:  ' + str(y.shape))
    print('W1: ' + str(W1.shape))
    print('W2: ' + str(W2.shape))

    costs = []
    running = True
    count = 0
    while running:
        running = False
        count += 1
        # Hidden layer

        # Z1 is 25xm
        Z1 = W1 * A[0]
        # tempA is 25xm
        tempA = vsig(Z1)
        # tempA is 26xm
        tempA = np.concatenate((np.ones((1,tempA.shape[1])),tempA), axis=0)
        # A1 is 26xm
        A.append(tempA)

        # Output layer

        # Z2 is 10xm
        Z2 = W2 * A[1]
        # A2 is 10xm
        A.append(vsig(Z2))

        # COST 
        J = funCosto(A[-1], y)
        print('IT #' + str(count) + ' J=' + str(J), end='\r')
        costs.append(J)
        #print('J=  ' + str(J))

        # BACK PROPAGATION

        # 10xm = (10xm - 10xm) .* 10xm
        d2 = np.multiply(A[-1] - y, sigmoidalGradienteA(A[2]))
        d2 = A[-1] - y

        # 26xm = (26x10 * 10xm) .* 26xm
        d1 = np.multiply(W2.T * d2, sigmoidalGradienteA(A[1]))

        # 25xm
        d1 = d1[1:]

        # PARAM UPDATE
        W2 -= alpha * d2 * A[1].T
        W1 -= alpha * d1 * A[0].T

        if (len(costs) < 100) or deltaIsBig(costs[-1], costs[-2]):
            running = True
        else:
            print('\nSTOP by Delta')
            pass
        if( len(costs) > 100 and costs[-1] > costs[-2]):
            running = False
            print('\nSTOP by Bounce')
        if count > 5000:
            running = False
            print('\nSTOP by Iterations')

    print('\nTRAINING DONE')
    return W1, W2, costs

def deltaIsBig(current, last):
    return abs(current-last) > 0.00000001


def sigmoidalGradiente(z):
    return np.multiply(vsig(z), 1-vsig(z))

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
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    data = np.reshape(data, (20,20))

    for x in range(0,20):
        for y in range(0, 20):
            color = (data[x,y]+1.0)/2.0
            ax.add_patch(
                patches.Rectangle(
                    (float(x), float(19-y)), 1.0, 1.0,
                    facecolor = (color, color, color)
                )
            )
    axes = plt.gca()
    axes.set_xlim([0,20])
    axes.set_ylim([0,20])
    plt.title('Drawing: ' + str(val))
    plt.show()

def training():
    print('\nREADING:')
    #X, y = readFile('digitos5.txt')
    X, y = readFile('digitos1000.txt')
    print('\nSHAPES:')
    print(X.shape)
    print(y.shape)
    #return
    print('\nTRAINING:')
    W1, W2, costs = entrenaRN(400, 25, 10, X, y)
    #print(W1)
    #print(W2)
    np.save('w1_1000_5.npy', W1)
    np.save('w2_1000_5.npy', W2)

    plt.plot(costs)
    plt.show()
    return W1, W2

def predictions(W1, W2):
    Xt, yt = readFile('digitos1000.txt')
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

    #W1 = np.load('w1_1000_4.npy')
    #W2 = np.load('w2_1000_4.npy')

    predictions(W1, W2)



if __name__ == '__main__':
    main()