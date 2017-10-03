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
    for num in range(0, 4):
        net[num,0] = escalon(net[num,0])
    cost = y - net
    return cost

def entrenaPerceptron(X, y, theta):
    running = True
    errors = []
    while running:
        running = False
        cost = funcionCostoPerceptron(theta,X,y)
        sumErr = 0
        for num in range(0, 4):
            error = cost[num,0]
            theta += error * X[num].T
            sumErr += error
            #if error != 0:
                #running = True
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
    s = X * theta
    error = y - s
    esquared = np.power(error, 2)
    return esquared, error

def entrenaAdaline(X, y, theta):
    running = True
    errors = []
    while running:
        running = False
        #costo, grad = funcionCostoAdaline(theta,X,y)
        lms = 0
        for num in range(0, 4):
            x = X[num]
            net = x * theta
            error = (y[num] - net).sum()
            sqrd = error * error
            lms += sqrd
            theta += 0.1 * error * x.T
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
    X, Y = readFile('and.txt')
    theta0 = np.matrix([1.5, 0.5, 1.5]).T
    theta1, errors = entrenaPerceptron(X, Y, theta0)
    print(theta1)
    print(predicePerceptron(theta1, [0, 0]))
    print(predicePerceptron(theta1, [0, 1]))
    print(predicePerceptron(theta1, [1, 0]))
    print(predicePerceptron(theta1, [1, 1]))

    theta0 = np.matrix([0.5, 0.5, 1.5]).T
    theta2, errors = entrenaAdaline(X, Y, theta0)
    #theta2 = np.matrix([[ 0.27777778],[ 0.44444444],[ 0.47222222]])
    plt.plot(errors)
    plt.show()
    print(theta2)
    print(prediceAdaline(theta2, [0, 0]))
    print(prediceAdaline(theta2, [0, 1]))
    print(prediceAdaline(theta2, [1, 0]))
    print(prediceAdaline(theta2, [1, 1]))
    