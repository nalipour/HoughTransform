import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
matplotlib.get_configdir()
print(plt.style.available)
plt.style.use('presentation')
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline


# def producePoints(a, b, offset, T, n):
#     # t = range(T)
#     t = np.linspace(0, T, n)
#     x = [a * np.cos(i) + offset for i in t]
#     y = [b * np.sin(i) + offset for i in t]
#     z = [b * i for i in t]
#     # df = pd.DataFrame({'x': x, 'y': y, 'z': z})
#     return x, y, z

def producePoints(x0, y0, z0, R, phi0, h, lamb, h_range, numpoints):

    t = np.linspace(0, h_range, numpoints)
    x = [x0 + R * (np.cos(phi0 + h * s * np.cos(lamb) / R) - np.cos(phi0)) for s in t]
    y = [y0 + R * (np.sin(phi0 + h * s * np.cos(lamb) / R) - np.sin(phi0)) for s in t]
    z = [z0 + s * np.sin(lamb) for s in t]
    return x, y, z


def conformalTransform(x, y):
    rhit_squared = (x**2 + y**2)
    if rhit_squared:
        xp = x / rhit_squared
        yp = y / rhit_squared
        # print("x=", x, ", xp=", xp, ", y=", y, ", yp=", yp, ", R=", rhit_squared)
        return xp, yp, rhit_squared
    else:
        return 0, 0, 0



def rho(rhit_squared, xp, yp, phi):
    # r = (2 / (rhit_squared)) * (xp * np.cos(phi) + yp * np.sin(phi))
    r = (xp * np.cos(phi) + yp * np.sin(phi))
    return r


def rho_phi(rhit_squared, xp, yp, numpoints):
    phis = np.linspace(0, np.pi, numpoints)
    rhos = [rho(rhit_squared, xp, yp, phi) for phi in phis]
    return phis, rhos


def HoughTransform_phi(Rsquared, Xp, Yp, numpoints, binx, biny, myrange):

    ht_phi = []
    ht_rho = []
    for i in range(0, len(Xp)):
        phis, rhos = rho_phi(Rsquared[i], Xp[i], Yp[i], numpoints)
        ht_phi.extend(phis)
        ht_rho.extend(rhos)

    print("Max R = ", max(ht_rho))
    fig_HT_phi = plt.figure()
    plt.hist2d(ht_phi, ht_rho, bins=(binx, biny), cmap=plt.cm.jet, range=myrange)
    plt.xlabel('phi')
    plt.ylabel('Rho')
    plt.tight_layout()
    plt.savefig('helix_hough.pdf')


def D_theta(x1, y1, x2, y2, numpoints):
    thetas = np.linspace(-2*np.pi, 2*np.pi, numpoints)
    D = [(1/2.) * ((y1**2 - y2**2 + x1**2 -x2**2) /((y1-y2)*np.sin(theta)+(x1 -
    x2)*np.cos(theta))) for theta in thetas]
    return thetas, D


def HT_D_theta(Xp, Yp, numpoints, binx, biny, myrange):
    ht_thetas = []
    ht_D = []
    for i in range(0, len(Xp)-1):
        t, d = D_theta(Xp[i], Yp[i], Xp[i+1], Yp[i+1], numpoints)
        ht_thetas.extend(t)
        ht_D.extend(d)

    fig_HT = plt.figure()
    plt.hist2d(ht_thetas, ht_D, bins=(binx, biny), cmap=plt.cm.jet, range=myrange)
    plt.xlabel('theta')
    plt.ylabel('D')
    plt.tight_layout()
    plt.savefig('helix_hough.pdf')
    return ht_thetas, ht_D


# X, Y, Z = producePoints(a=1, b = 1, offset = np.sqrt(2), T = 3 * np.pi / 2., n=100)
# #df.head()
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X, Y, Z, c = 'DarkBlue', linestyle='-')
# ax.plot(X, Y, Z, color='b')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.tight_layout()
# plt.savefig("helix.pdf")
#
# fig2d = plt.figure()
# plt.scatter(X, Y, c = 'DarkBlue', linestyle='-')
#
# XY_p = [conformalTransform(X[i], Y[i]) for i in range(0, len(X))]
# Xp, Yp, rhit_squared = zip(*XY_p)
#
# print(rhit_squared)
#
# fig2 = plt.figure()
# plt.scatter(Xp, Yp, c = 'DarkBlue', linestyle='-')
# plt.xlabel('xp')
# plt.ylabel('yp')
# plt.tight_layout()
# plt.savefig("ConformalTransform.pdf")
#
#
# rhos, phis = rho_phi(rhit_squared[0], Xp[0], Yp[0], 100)
# fig3 = plt.figure()
# plt.scatter(phis, rhos, c = 'DarkBlue', linestyle='-')
# plt.xlabel('p')
# plt.ylabel('R')
# plt.tight_layout()
# plt.savefig("phi_rho.pdf")
#
#
# HoughTransform_phi(Rsquared=rhit_squared, Xp=Xp, Yp=Yp, numpoints=100, binx=100, biny = 100)
