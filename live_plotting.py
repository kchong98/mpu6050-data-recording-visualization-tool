import random
import numpy as np
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def getdisplacement(data):
    x_list = np.array([]).reshape(0, 1)
    v01 = 0
    x0 = 0
    for i in range(2, len(data)):
        v0 = (1/2)*(data[i-1]+data[i-2])*0.2+v01
        v01 = v0
        xf = (1/4)*(data[i]-data[i-1])*0.04+v0*0.2+x0
        x0 = xf
        x_list = np.vstack((x_list, x0))
    return x_list


fig = plt.figure()
ax = fig.gca(projection='3d')

plt.style.use('fivethirtyeight')

index = count()

def animate(i):
    data = pd.read_csv('<Name of the csv file>.csv') # Name of the CSV file needs to be configured
    data = pd.DataFrame.to_numpy(data)
    data = data[1:, 3:6]

    X = data[:, 0].astype(np.float)
    Y = data[:, 1].astype(np.float)
    Z = data[:, 2].astype(np.float)
    x_displacement = getdisplacement(X)
    y1 = x_displacement.reshape(len(x_displacement), )
    y2 = getdisplacement(Y).reshape(len(x_displacement), )
    y3 = getdisplacement(Z).reshape(len(x_displacement), )

    plt.cla()
    plt.plot(y1, y2, y3, label='Random 3D data')
    plt.legend(loc='upper left')
    plt.tight_layout()

ani = FuncAnimation(plt.gcf(), animate, interval=50)

plt.tight_layout()
plt.show()