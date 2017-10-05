import csv
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
            xIn[-1].append(float(row[0]))
            xIn[-1].append(float(row[1]))
            yIn.append(int(row[2]))
    print('Reading OK!')
    xIn = np.matrix(xIn)
    yIn = np.matrix(yIn).T
    return xIn, yIn

def escalon(x):
    if x < 0:
        return 0
    else:
        return 1

def funcionCostoPerceptron(theta,X,y):
    net = X * theta
    out = escalon(net)
    cost = y - out
    return cost

def entrenaPerceptron(X, y, theta):
    running = True
    errors = []
    while running:
        if len(errors) > 5000:
            print('ERROR')
            break;
        running = False
        sumErr = 0
        for num in range(0, 4):
            error = funcionCostoPerceptron(theta, X[num], y[num]).sum()
            theta += error * X[num].T
            sumErr += abs(error)
        errors.append(sumErr/4)
        if sumErr != 0:
           running = True

    return theta, errors

def predicePerceptron(theta, X):
    X = np.matrix(X)
    X = np.matrix(np.concatenate(([1], X.A1)))
    net = X * theta
    return escalon(net)

def escalonAda(x):
    if x < 0.5:
        return 0
    else:
        return 1

def funcionCostoAdaline(theta,X,y):
    net = X * theta
    error = (y - net).sum()
    sqrd = error * error

    return sqrd, error

def entrenaAdaline(X, y, theta):
    running = True
    errors = []
    while running:
        if len(errors) > 5000: 
            print('ERROR')
            break;
        running = False
        lms = 0
        for num in range(0, 4):
            costo, grad = funcionCostoAdaline(theta,X[num],y[num])
            lms += costo
            theta += 0.1 * grad * X[num].T
        lms /= 8
        if len(errors) == 0 or deltaIsBig(lms, errors[-1]):
            running = True
        errors.append(lms)
    return theta, errors

def deltaIsBig(current, last):
    return abs(current-last) > 0.0000001

def prediceAdaline(theta, X):
    X = np.matrix(X)
    X = np.matrix(np.concatenate(([1], X.A1)))
    net = X * theta
    return escalonAda(net)

if __name__ == '__main__':
    X = np.matrix([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    Ys = {'1':np.matrix([0,0,0,1]).T, '2':np.matrix([0,1,1,1]).T, '3':np.matrix([0,1,1,0]).T}
    opt = {'1':'and', '2':'or', '3':'xor'}
    print('Select:\n  1 - AND\n  2 - OR\n  3 - XOR')

    sel = input()
    Y = Ys[sel]
    target = opt[sel]

    print('APRENDIENDO: ' + target.upper())
    print('\nPerceptrón\nW=')
    theta0 = np.matrix([1.5, 1.5, 1.5]).T
    theta1, errors = entrenaPerceptron(X, Y, theta0)
    print(theta1)
    print('\n a b | a ' + target + ' b')
    print(' 0 0 | ', end='')
    print(predicePerceptron(theta1, [0, 0]))
    print(' 0 1 | ', end='')
    print(predicePerceptron(theta1, [0, 1]))
    print(' 1 0 | ', end='')
    print(predicePerceptron(theta1, [1, 0]))
    print(' 1 1 | ', end='')
    print(predicePerceptron(theta1, [1, 1]))
    plot1, = plt.plot(errors, 'r', label='Perceptron')

    print('\nAdaline:\nW=')
    theta0 = np.matrix([0.5, 0.5, 1.5]).T
    theta2, errors = entrenaAdaline(X, Y, theta0)
    print(theta2)
    print('\n a b | a ' + target + ' b')
    print(' 0 0 | ', end='')
    print(prediceAdaline(theta2, [0, 0]))
    print(' 0 1 | ', end='')
    print(prediceAdaline(theta2, [0, 1]))
    print(' 1 0 | ', end='')
    print(prediceAdaline(theta2, [1, 0]))
    print(' 1 1 | ', end='')
    print(prediceAdaline(theta2, [1, 1]))
    plot2, = plt.plot(errors, 'b', label='Adaline')
    plt.ylabel('Error')
    plt.xlabel('Iteración')
    plt.legend(handles=[plot1, plot2])
    plt.show()
    