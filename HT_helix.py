import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
matplotlib.get_configdir()
print(plt.style.available)
plt.style.use('presentation')
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline


def producePoints(a, b, offset, T, n):
    # t = range(T)
    t = np.linspace(0, T, n)
    x = [a * np.cos(i) + offset for i in t]
    y = [b * np.sin(i) + offset for i in t]
    z = [b * i for i in t]
    # df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    return x, y, z


def conformalTransform(x, y):
    xp = x / (x**2 + y**2)
    yp = y / (x**2 + y**2)
    rhit_squared = (x**2 + y**2)

    return xp, yp, rhit_squared


def rho(rhit_squared, xp, yp, phi):
    r = (2 / (rhit_squared)) * (xp * np.cos(phi) + yp * np.sin(phi))
    return r


def rho_phi(rhit_squared, xp, yp, numpoints):
    phis = np.linspace(0, np.pi, numpoints)
    rhos = [rho(rhit_squared, xp, yp, phi) for phi in phis]
    return phis, rhos


def HoughTransform_phi(Rsquared, Xp, Yp, numpoints, binx, biny):

    ht_phi = []
    ht_rho = []
    for i in range(0, len(Xp)):
        phis, rhos = rho_phi(Rsquared[i], Xp[i], Yp[i], numpoints)
        ht_phi.extend(phis)
        ht_rho.extend(rhos)
        #HT = [rho_phi(rhit_squared, xp, yp, numpoints) for rhit_squared, xp, yp in zip(Rsquared, Xp, Yp)]
    #H, xedges, yedges = np.histogram2d(ht_phi, ht_rho, bins = (binx, biny))
    print("Max R = ", max(ht_rho))
    plt.hist2d(ht_phi, ht_rho, bins=(binx, biny), cmap=plt.cm.jet)
    plt.xlabel('p')
    plt.ylabel('R')
    plt.tight_layout()
    plt.savefig('helix_hough.pdf')





X, Y, Z = producePoints(a=1, b = 1, offset = np.sqrt(2), T = 3 * np.pi / 2., n=100)
#df.head()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, c = 'DarkBlue', linestyle='-')
ax.plot(X, Y, Z, color='b')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.tight_layout()
plt.savefig("helix.pdf")

fig2d = plt.figure()
plt.scatter(X, Y, c = 'DarkBlue', linestyle='-')

XY_p = [conformalTransform(X[i], Y[i]) for i in range(0, len(X))]
Xp, Yp, rhit_squared = zip(*XY_p)

print(rhit_squared)

fig2 = plt.figure()
plt.scatter(Xp, Yp, c = 'DarkBlue', linestyle='-')
plt.xlabel('xp')
plt.ylabel('yp')
plt.tight_layout()
plt.savefig("ConformalTransform.pdf")


rhos, phis = rho_phi(rhit_squared[0], Xp[0], Yp[0], 100)
fig3 = plt.figure()
plt.scatter(phis, rhos, c = 'DarkBlue', linestyle='-')
plt.xlabel('p')
plt.ylabel('R')
plt.tight_layout()
plt.savefig("phi_rho.pdf")


HoughTransform_phi(Rsquared=rhit_squared, Xp=Xp, Yp=Yp, numpoints=100, binx=100, biny = 100)
