import random
import numpy as np
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# data=pd.read_csv("SensorDataFile2.csv", header=None)
# data = pd.DataFrame.to_numpy(data)
# data = data[1:, 3:6]

# X = data[:, 0].astype(np.float)
# Y = data[:, 1].astype(np.float)
# Z = data[:, 2].astype(np.float)


# def getdisplacement(data):
#     x_list = np.array([]).reshape(0, 1)
#     v01 = 0
#     x0 = 0
#     for i in range(2, len(data)):
#         v0 = (1/2)*(data[i-1]+data[i-2])*0.2+v01
#         v01 = v0
#         xf = (1/4)*(data[i]-data[i-1])*0.02+v0*0.2+x0
#         x0 = xf
#         x_list = np.vstack((x_list, x0))
#     return x_list
    
# x_displacement = getdisplacement(X)
# x_displacement = x_displacement.reshape(len(x_displacement), )
# y_displacement = getdisplacement(Y).reshape(len(x_displacement), )
# z_displacement = getdisplacement(Z).reshape(len(x_displacement), )

fig = plt.figure()
ax = fig.gca(projection='3d')

plt.style.use('fivethirtyeight')

index = count()

def animate(i):
    data = pd.read_csv('data.csv')
    x = data['x_value']
    y1 = data['total_1']
    y2 = data['total_2']
    y3 = data['total_3']

    plt.cla()
    plt.plot(y1, y2, y3, label='Random 3D data')
    plt.legend(loc='upper left')
    plt.tight_layout()

ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()