from HT_helix import conformalTransform, HoughTransform_phi, HT_D_theta
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
matplotlib.get_configdir()
print(plt.style.available)
plt.style.use('seaborn-poster')
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline


def produceCircle(a, b, R, n, T):
    t = np.linspace(0, T, n)
    x = [a + R * np.cos(i) for i in t]
    y = [b + R * np.sin(i) for i in t]

    return x, y

def conformalTransform(x, y):
    rhit_squared = (x**2 + y**2)
    if rhit_squared:
        xp = 2 * x / rhit_squared
        yp = 2 * y / rhit_squared
        return xp, yp, rhit_squared
    else:
        return 0, 0, 0

radius = 20
_a = 5
_b = np.sqrt(radius**2-_a**2)
print(_b)

radius2 = 40
_a2 = 5
_b2 = np.sqrt(radius2**2-_a2**2)

X1, Y1 = produceCircle(a = _a, b = _b, R = radius, n = 20, T = 1 * np.pi)
print("1: len: ", len(X1))
X2, Y2 = produceCircle(a = _a2, b = _b2, R = radius2, n = 20, T = 1 * np.pi)
X = X1
Y = Y1
X.extend(X2)
Y.extend(Y2)

print("2: len: ", len(X2))

figCirc= plt.figure()
plt.scatter(X1[:len(X2)], Y1[:len(X2)], c = 'DarkBlue', linestyle='-', label='Track1')
plt.scatter(X2, Y2, c = 'red', linestyle='-', label='Track2')
plt.legend(loc='upper left');
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
plt.savefig('plots/tracksXY.pdf')


XY_p = [conformalTransform(X[i], Y[i]) for i in range(0, len(X)) if
        conformalTransform(X[i], Y[i])[2]]
Xp, Yp, rhit_squared = zip(*XY_p)
fig_conformalTransform = plt.figure()
plt.scatter(Xp[:len(X2)], Yp[:len(X2)], c = 'DarkBlue', linestyle='-', label='Track1')
plt.scatter(Xp[len(X2):], Yp[len(X2):], c = 'red', linestyle='-', label='Track2')
plt.title("Conformal transform")
plt.legend(loc='upper left')
plt.xlabel("u")
plt.ylabel("v")
plt.tight_layout()
plt.savefig('plots/tracksConformalTransform.pdf')


R_, theta_ = HoughTransform_phi(Rsquared=rhit_squared, Xp=Xp, Yp=Yp,
                                numpoints=500,
                                binx=200, biny = 200, myrange=[[-0*np.pi, np.pi], [0, 0.1]])

# print (rhit_squared)
# print(1/(radius))
radius_calc = 1./(R_)
a_calc = np.cos(theta_)/(R_)
b_calc = np.sqrt(radius_calc**2 - a_calc**2)

print("a =", a_calc)
print("b =", b_calc)
print("radius=", radius_calc)
