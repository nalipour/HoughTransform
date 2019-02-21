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
# energy="_1MeV"
# Filename = "mergedHits_2T"+energy+".csv"
# FilenameG4hits = "G4Hits_2T"+energy+".csv"

energy="_1MeV"
Filename = "mergedHits_2T"+energy+".csv"
FilenameG4hits = "G4Hits_2T"+energy+".csv"
Filename_background = "reco_0.csv"

def MMtoMeter(df, vars):
    for i in vars:
        df[i] = df[i]/1000.

def creatRandBackground(numwires, radius, halfLength):
    x = np.random.uniform(low=-radius, high=radius, size=(numwires))
    y = np.random.uniform(low=-radius, high=radius, size=(numwires))
    z = np.random.uniform(low=-halfLength, high=halfLength, size=(numwires))

    data_tuples = list(zip(x, y, z))
    return pd.DataFrame(data_tuples, columns=['MCx', 'MCy', 'MCz'])

csv_path = os.path.join(DATA_PATH, Filename)
csv_path2 = os.path.join(DATA_PATH, FilenameG4hits)
csv_path_bcg = os.path.join(DATA_PATH, Filename_background)
# MC hit position (intersection between the wire and the track)
dataset = pd.read_csv(csv_path)
# Positions of the hits at each G4 step
g4hits = pd.read_csv(csv_path2)
# Background
background = pd.read_csv(csv_path_bcg)
# background = creatRandBackground(int(0.001*56000), 1700, 2000)

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

# Background + hits
hitsBcg = pd.concat([hits, background], ignore_index=True)

# resx = plt.figure()
# n, bins, patches = plt.hist(x=dataset["MCx"]-hits["MCx"], bins=100,
#                             alpha=0.7, rwidth=0.85)
#
# plt.title("Hit resolution")
# plt.xlabel("$MC_x$ - $Reco_x$ [mm]")
# plt.ylabel("Entries")
# plt.tight_layout()
#
# resy = plt.figure()
# n, bins, patches = plt.hist(x=dataset["MCy"]-hits["MCy"], bins=100,
#                             alpha=0.7, rwidth=0.85)
#
# plt.title("Hit resolution")
# plt.xlabel("$MC_y$ - $Reco_y$ [mm]")
# plt.ylabel("Entries")
# plt.tight_layout()
#
# resz = plt.figure()
# n, bins, patches = plt.hist(x=dataset["MCz"]-hits["MCz"], bins=100,
#                             alpha=0.7, rwidth=0.85)
#
# plt.title("Hit resolution")
# plt.xlabel("$MC_z$ - $Reco_z$ [mm]")
# plt.ylabel("Entries")
# plt.tight_layout()


# ## MM to Meter
# MMtoMeter(g4hits, ["x", "y", "z"])
# MMtoMeter(dataset, ["MCx", "MCy", "MCz"])
# MMtoMeter(hits, ["MCx", "MCy", "MCz"])

# Plot the raw hits
figHits = plt.figure()
ax = Axes3D(figHits)
# ax.scatter(g4hits['x'], g4hits['z'], g4hits['y'], c = 'blue', marker='x', label="G4 steps")
ax.scatter(dataset['MCx'], dataset['MCz'], dataset['MCy'], c = 'red', marker = 'o', label="MC Truth")
ax.scatter(hits['MCx'], hits['MCz'], hits['MCy'], c = 'green', marker = '*', label="Reco Hit")
ax.scatter(background['MCx'], background['MCz'], background['MCy'], c = 'blue', marker = '*', label="Background")
plt.legend(loc='upper left')
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_ylabel("y")
plt.tight_layout()
ax.xaxis.labelpad = 20
ax.yaxis.labelpad = 20
ax.zaxis.labelpad = 20
plt.show()


fig_xy = plt.figure()
# plt.scatter(g4hits['x'], g4hits['y'], c = 'blue', marker='x', label="G4 steps")
plt.scatter(dataset['MCx'], dataset['MCy'], c = 'red', marker = '+', label="MC Truth")
plt.scatter(hits['MCx'], hits['MCy'], c = 'green', marker = '*', label="Reco Hit")
plt.scatter(background['MCx'], background['MCy'], c = 'blue', marker = '*', label="Background")
plt.legend(loc='upper left')
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.tight_layout()
plt.show()

# plt.savefig('plots/bcg_hits_xy_3percent.pdf')

# fig_xz = plt.figure()
# plt.scatter(g4hits['x'], g4hits['z'], c = 'blue', marker='x', label="G4 steps")
# plt.scatter(dataset['MCx'], dataset['MCz'], c = 'red', marker = '+', label="MC Truth")
# plt.scatter(hits['MCx'], hits['MCz'], c = 'green', marker = '*', label="Reco Hit")
# plt.legend(loc='upper left')
# plt.xlabel('x')
# plt.ylabel('z')
# plt.tight_layout()
#
# fig_yz = plt.figure()
# plt.scatter(g4hits['y'], g4hits['z'], c = 'blue', marker='x', label="G4 steps")
# plt.scatter(dataset['MCy'], dataset['MCz'], c = 'red', marker = '+', label="MC Truth")
# plt.scatter(hits['MCy'], hits['MCz'], c = 'green', marker = '*', label="Reco Hit")
# plt.legend(loc='upper left')
# plt.xlabel('y')
# plt.ylabel('z')
# plt.tight_layout()
#
# plt.show()


# DoFullHT(hits['MCx'], hits['MCy'], mrange=[[0, np.pi], [-0.0005, 0.0005]],
#          numpoints=500, binx=200, biny = 200, rangeConf=[(0.0002, 0.003), (0.00011, 0.00013)], plotName = "2400MeV")

DoFullHT(hitsBcg['MCx'], hitsBcg['MCy'], mrange=[[0, np.pi], [-0.003, 0.003]],
         numpoints=500, binx=200, biny = 200, rangeConf=[(-0.003, 0.003), (-0.003, 0.003)], plotName = "backgroundHit_3percent")
