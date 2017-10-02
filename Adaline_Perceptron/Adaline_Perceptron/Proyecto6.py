import csv
import numpy as np

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
    pass

def entrenaPerceptron(X, y, theta):
    running = True
    while running:
        running = False
        for num in range(0, 4):
            x = X[num]
            net = x * theta
            ev = escalon(net)
            error = y[num,0] - ev
            theta += error * x.T
            if error != 0:
                running = True

    return theta

def predicePerceptron(theta, X):
    X = np.matrix(X)
    X = np.matrix(np.concatenate(([1], X.A1)))
    net = X * theta
    return escalon(net)

def funcionCostoAdaline(theta,X,y):
    s = X * theta
    error = y - s
    esquared = np.power(error, 2)
    return esquared, error

def entrenaAdaline(X, y, theta):
    running = True
    while running:
        running = False
        lms = 0
        for num in range(0, 4):
            x = X[num]
            net = x * theta
            ev = escalon(net)
            error = y[num,0] - ev
            lms += error * error
            theta += 0.01*error * x.T
        lms /= (2*4)
        print(lms, end="\r")
        if lms > 0.01:
            running = True

    return theta

def entrenaAdaline2(X, y, theta):
    running = True
    lmss = []
    while running:
        running = False
        print("theta:")
        print(theta)
        costo, error = funcionCostoAdaline(theta, X, y)

        for num in range(0, 4):
            x = X[num]
            e = error[num,0]
            theta += 0.1 * e * x.T
        lms = costo.sum() / ( 2 * 4 )
        print(lms, end="\r")
        if(len(lmss) != 0):
            print(str(lms) + " - " + str(lmss[-1]) + " = " + str((lms - lmss[-1])))
        if len(lmss) == 0 or abs(lms - lmss[-1]) > 0.001:
            running = True
        lmss.append(lms)

    return theta

def prediceAdaline(theta, X):
    X = np.matrix(X)
    X = np.matrix(np.concatenate(([1], X.A1)))
    net = X * theta
    return escalon(net)

if __name__ == '__main__':
    X, Y = readFile('and.txt')
    theta0 = np.matrix([1.5, 0.5, 1.5]).T
    theta1 = entrenaPerceptron(X, Y, theta0)
    print(theta1)
    print(predicePerceptron(theta1, [0, 0]))
    print(predicePerceptron(theta1, [0, 1]))
    print(predicePerceptron(theta1, [1, 0]))
    print(predicePerceptron(theta1, [1, 1]))

    theta0 = np.matrix([1.5, 0.5, 1.5]).T
    theta2 = entrenaAdaline(X, Y, theta0)
    print(theta2)
    print(prediceAdaline(theta2, [0, 0]))
    print(prediceAdaline(theta2, [0, 1]))
    print(prediceAdaline(theta2, [1, 0]))
    print(prediceAdaline(theta2, [1, 1]))
    