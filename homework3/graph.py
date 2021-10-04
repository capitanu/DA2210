import matplotlib.pyplot as plt
import matplotlib

from sklearn import linear_model


import numpy as np
import math

def readlines():
    txt = open("data.txt", "r")
    data = txt.readlines()
    for i in range(len(data)):
        data[i] = data[i].strip('\n')
        data[i] = data[i].strip('\t')
        data[i] = data[i].split("\t")
        data[i].append(i)
        data[i][1] = int(data[i][1])
    data = np.array(data)
    return data.T


def graph():
    data = readlines()
    x = np.log10(data[2].astype(int))
    y = np.log10(data[1].astype(int))
    plt.plot(x, y)
    plt.ylabel("Frequency")
    plt.xlabel("Rank")
    plt.title("Frequency of English words according to their rank in a log-log graph")
    plt.savefig('histogram.png', dpi=300)
    plt.show()

def lr():
    data = readlines()
    x = np.log10(data[2].astype(int).reshape(-1,1))
    y = np.log10(data[1].astype(int).reshape(-1,1))
    x = x[1:]
    y = y[1:]
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    pred = regr.predict(x)
    
    print('Coefficients: \n', math.pow(10, regr.coef_))
    plt.ylabel("Frequency")
    plt.xlabel("Rank")
    plt.title("Linear Regression in a log-log graph")

    plt.scatter(x,y)
    plt.plot(x, pred, c='r')
    plt.savefig('lr.png', dpi=200)    
    plt.show()
    
graph()
lr()
