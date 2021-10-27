import numpy as np
import matplotlib.pyplot as plt

T = np.load('speedtest.npy')
X = []
y1 = []
y2 = []
for i in range(len(T)):
    X.append(T[i][0])
    y1.append(T[i][1])
    y2.append(T[i][3])

plt.plot(X,y1)
plt.show()
