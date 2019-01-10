import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
matplotlib.get_configdir()
print(plt.style.available)
plt.style.use('presentation')

from matplotlib import interactive
interactive(True)

%matplotlib inline

def producePoints(a, b, n, numpoints):
    x = np.linspace(0, n, numpoints)
    y = [a * i + b for i in x]
    df = pd.DataFrame({'x': x, 'y': y})
    return df

def HoughTransform(x, y, n, binx, biny):

    theta = np.linspace(0, np.pi, n)

    t_arr = []
    r_arr = []
    for t in theta:
        for i in range(0, len(x)):
            r = x[i] * np.cos(t) + y[i] * np.sin(t)
            t_arr.append(t)
            r_arr.append(r)
        # R = np.multiply(x, np.cos(t)) + np.multiply(y, np.sin(t))
    # ret = stats.binned_statistic_2d(t_arr, r_arr, None, 'count',
    #                                 statistic='max', bins=[binx, biny])
    H, xedges, yedges = np.histogram2d(t_arr, r_arr, bins = (binx, biny))
    #H = H.T
    print(xedges)
    print("size=", len(xedges))
    print("max=", H.max())
    print("amax=", H.argmax())
    am = H.argmax()
    r_idx = am % H.shape[1]
    c_idx = am // H.shape[1]

    print("x index = ", xedges[c_idx])
    print("y index = ", yedges[r_idx])

    R_ = yedges[r_idx]
    theta_ = xedges[c_idx]
    print("Result: ", (R_ - 5 * np.cos(theta_))/np.sin(theta_))
    print("b=", R_ * np.cos(theta_))

    #plt.imshow(H, origin='low', cmap=plt.cm.jet)#, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.hist2d(t_arr, r_arr, bins=(binx, biny), cmap=plt.cm.jet)
    plt.xlabel(r'$\theta$')
    plt.ylabel('R')
    plt.tight_layout()
    plt.savefig('line_hough.pdf')


npoints = 2000

df = producePoints(a=1, b=2, n=10, numpoints=npoints)
df.head()
df.plot.scatter(x = 'x', y = 'y', c = 'DarkBlue')
plt.tight_layout()
plt.savefig("line.pdf")
np.pi
theta = np.linspace(0., np.pi, 4)
r = theta * 2
np.max(theta)
np.multiply(list(df["x"]), 2.0)

HoughTransform(list(df["x"]), list(df["y"]), npoints, 50, 50)
