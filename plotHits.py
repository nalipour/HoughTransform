from ConformalTransform import conformalTransform, HoughTransform_phi, DoFullHT
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# matplotlib.get_configdir()
print(plt.style.available)
plt.style.use('seaborn-poster')
%matplotlib inline

DATA_PATH = './data'
energy="_1MeV"
Filename = "mergedHits_2T"+energy+".csv"
FilenameG4hits = "G4Hits_2T"+energy+".csv"

def MMtoMeter(df, vars):
    for i in vars:
        df[i] = df[i]/1000.

csv_path = os.path.join(DATA_PATH, Filename)
csv_path2 = os.path.join(DATA_PATH, FilenameG4hits)
# MC hit position (intersection between the wire and the track)
dataset = pd.read_csv(csv_path)
# Positions of the hits at each G4 step
g4hits = pd.read_csv(csv_path2)


# Resolution in x, y, z
hits = dataset.copy()
sigma_xy = 0.1 # mm
sigma_z = 1 # mm

randx = np.random.uniform(low=-1*sigma_xy, high=sigma_xy, size=(hits.shape[0]))
randy = np.random.uniform(low=-1*sigma_xy, high=sigma_xy, size=(hits.shape[0]))
randz = np.random.uniform(low=-1*sigma_z, high=sigma_z, size=(hits.shape[0]))

# smeared hits
hits["MCx"] += randx
hits["MCy"] += randy
hits["MCz"] += randz


resx = plt.figure()
n, bins, patches = plt.hist(x=dataset["MCx"]-hits["MCx"], bins=100,
                            alpha=0.7, rwidth=0.85)

plt.title("Hit resolution")
plt.xlabel("$MC_x$ - $Reco_x$ [mm]")
plt.ylabel("Entries")
plt.tight_layout()

resy = plt.figure()
n, bins, patches = plt.hist(x=dataset["MCy"]-hits["MCy"], bins=100,
                            alpha=0.7, rwidth=0.85)

plt.title("Hit resolution")
plt.xlabel("$MC_y$ - $Reco_y$ [mm]")
plt.ylabel("Entries")
plt.tight_layout()

resz = plt.figure()
n, bins, patches = plt.hist(x=dataset["MCz"]-hits["MCz"], bins=100,
                            alpha=0.7, rwidth=0.85)

plt.title("Hit resolution")
plt.xlabel("$MC_z$ - $Reco_z$ [mm]")
plt.ylabel("Entries")
plt.tight_layout()


## MM to Meter
MMtoMeter(g4hits, ["x", "y", "z"])
MMtoMeter(dataset, ["MCx", "MCy", "MCz"])
MMtoMeter(hits, ["MCx", "MCy", "MCz"])

# Plot the raw hits
figHits = plt.figure()
ax = Axes3D(figHits)
# ax.scatter(g4hits['x'], g4hits['z'], g4hits['y'], c = 'blue', marker='x', label="G4 steps")
ax.scatter(dataset['MCx'], dataset['MCz'], dataset['MCy'], c = 'red', marker = 'o', label="MC Truth")
ax.scatter(hits['MCx'], hits['MCz'], hits['MCy'], c = 'green', marker = '*', label="Reco Hit")
plt.legend(loc='upper left')
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_ylabel("y")
plt.tight_layout()
ax.xaxis.labelpad = 20
ax.yaxis.labelpad = 20
ax.zaxis.labelpad = 20


fig_xy = plt.figure()
plt.scatter(g4hits['x'], g4hits['y'], c = 'blue', marker='x', label="G4 steps")
plt.scatter(dataset['MCx'], dataset['MCy'], c = 'red', marker = '+', label="MC Truth")
plt.scatter(hits['MCx'], hits['MCy'], c = 'green', marker = '*', label="Reco Hit")
plt.legend(loc='upper left')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()

fig_xz = plt.figure()
plt.scatter(g4hits['x'], g4hits['z'], c = 'blue', marker='x', label="G4 steps")
plt.scatter(dataset['MCx'], dataset['MCz'], c = 'red', marker = '+', label="MC Truth")
plt.scatter(hits['MCx'], hits['MCz'], c = 'green', marker = '*', label="Reco Hit")
plt.legend(loc='upper left')
plt.xlabel('x')
plt.ylabel('z')
plt.tight_layout()

fig_yz = plt.figure()
plt.scatter(g4hits['y'], g4hits['z'], c = 'blue', marker='x', label="G4 steps")
plt.scatter(dataset['MCy'], dataset['MCz'], c = 'red', marker = '+', label="MC Truth")
plt.scatter(hits['MCy'], hits['MCz'], c = 'green', marker = '*', label="Reco Hit")
plt.legend(loc='upper left')
plt.xlabel('y')
plt.ylabel('z')
plt.tight_layout()

plt.show()

DoFullHT(dataset['MCx'], dataset['MCy'], mrange=[[1.4, 1.8], [-0, 0.2]],
         numpoints=100, binx=10, biny = 10)
# Conformal Transform
# XY_p = [conformalTransform(hits['MCx'][i], hits['MCy'][i])
#         for i in range(0, hits.shape[0]) if
#         conformalTransform(hits['MCx'][i], hits['MCy'][i])[2]]
# Xp, Yp, rhit_squared = zip(*XY_p)
# fig_conformalTransform = plt.figure()
# plt.scatter(Xp, Yp, c = 'DarkBlue', linestyle='-')
# plt.title("Conformal transform")
# plt.legend(loc='upper left')
# plt.xlabel("u")
# plt.ylabel("v")
# plt.tight_layout()
#
# R_, theta_ = HoughTransform_phi(Rsquared=rhit_squared, Xp=Xp, Yp=Yp,
#                                 numpoints=500,
#                                 binx=200, biny = 200, myrange=[[1., 2.], [-1, 1]])
#
# # print (rhit_squared)
# # print(1/(radius))
# radius_calc = 1./(2*R_)
# a_calc = np.cos(theta_)/(2*R_)
# b_calc = np.sin(theta_)/(2*R_)
#
# print("a =", a_calc)
# print("b =", b_calc)
# print("radius=", radius_calc)
