import random
import numpy as np
import imageio
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse



def findClosestCentroids(X, initial_centroids):
    d = distance.cdist(X, initial_centroids, 'sqeuclidean')
    closest = np.argmin(d, axis=1)
    return closest

def computeCentroids(X, idx, K):
    newK = np.zeros((K,X.shape[1]))
    adders = np.zeros(K)
    #print(adders.shape)
    for index, val in enumerate(X):
        centroid = idx[index]
        newK[centroid] += val
        adders[centroid] += 1
    adders = adders.reshape((len(adders),1))

    with np.errstate(divide='ignore', invalid='ignore'):
        out = newK / adders
        out = np.nan_to_num(out)

    return out



def runkMeans(X, initial_centroids, max_iters, draw=True):
    newK = initial_centroids
    c_last = None
    for it in range(max_iters):
        print("#" + str(it), end='\r')
        closest = findClosestCentroids(X, newK)
        newK = computeCentroids(X, closest, len(initial_centroids))
        c_curr = newK.sum()
        if c_last and not deltaIsBig(c_last, c_curr):
            break
        c_last = c_curr
    print("Done! \nIn " + str(it) + " iterations.")
    return newK

def deltaIsBig(current, last):
    return np.abs(current-last) > 0.00000001

def kMeansInitCentroids(X, K):
    m = len(X)
    Ks = []
    for i in range(K):
        Ks.append(X[random.randint(0,m-1)])
    return np.array(Ks)

def replaceCentroids(X, centroids):
    closest = findClosestCentroids(X, centroids)
    for index, val in enumerate(X):
        X[index] = centroids[closest[index]]
    return X


def main(picture='bird_small.png', colors=16):

    pic = np.array(imageio.imread(picture)).astype('uint8')

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(pic)
    plt.title('Original')
    plt.show(block=False)

    flat = pic.reshape((pic.shape[0]*pic.shape[1], pic.shape[2]))
    kInit = kMeansInitCentroids(flat,colors)
    newK = runkMeans(flat, kInit, 500, False)

    newPic = replaceCentroids(flat, newK).reshape((pic.shape[0], pic.shape[1], pic.shape[2]))

    plt.subplot(122)
    plt.imshow(newPic)
    plt.title('Reduced to ' + str(colors) + ' colors.')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image color compression.')
    parser.add_argument('-i', default='bird_small.png', help='Image to compress. (file name)')
    parser.add_argument('-c', type=int, default=16, help='Target number of colors. (integer)')
    args = parser.parse_args()
    main(args.i, args.c)
