import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model


with open("data.txt", "r") as inputFile:
    words = [tuple(line.split()) for line in inputFile.readlines()]

frequency = []
for tuple1 in words:
    frequency.append(tuple1[1])

frequency_length = len(frequency)
rank = (range(1, frequency_length + 1))


'''
regr = linearmodel.LinearRegression()
    regr.fit(x, y)
    pred = regr.predict(x)

regr.coef
'''


plt.scatter(np.log10(rank), np.log10(np.array(frequency).astype(int)))
plt.xlabel('frequency(f)', fontsize=14, fontweight='bold')
plt.ylabel('rank(r)', fontsize=14, fontweight='bold')
plt.title("English Language, Rank vs Frequency")
plt.grid(True)
plt.show()
