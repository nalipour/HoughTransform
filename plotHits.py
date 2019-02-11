import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# matplotlib.get_configdir()
print(plt.style.available)
plt.style.use('seaborn-poster')
%matplotlib inline

DATA_PATH = './data'
Filename = "mergedHits_2T.csv"
FilenameG4hits = "G4Hits_2T.csv"

csv_path = os.path.join(DATA_PATH, Filename)
csv_path2 = os.path.join(DATA_PATH, FilenameG4hits)
# MC hit position (intersection between the wire and the track)
dataset = pd.read_csv(csv_path)
# Positions of the hits at each G4 step
g4hits = pd.read_csv(csv_path2)



# Plot the raw hits
figHits = plt.figure()
ax = Axes3D(figHits)
ax.scatter(g4hits['x'], g4hits['z'], g4hits['y'], c = 'blue', marker='x', label="G4 steps")
ax.scatter(dataset['MCx'], dataset['MCz'], dataset['MCy'], c = 'red', marker = 'o', label="MC Truth")
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
plt.legend(loc='upper left')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()

fig_xz = plt.figure()
plt.scatter(g4hits['x'], g4hits['z'], c = 'blue', marker='x', label="G4 steps")
plt.scatter(dataset['MCx'], dataset['MCz'], c = 'red', marker = '+', label="MC Truth")
plt.legend(loc='upper left')
plt.xlabel('x')
plt.ylabel('z')
plt.tight_layout()

fig_yz = plt.figure()
plt.scatter(g4hits['y'], g4hits['z'], c = 'blue', marker='x', label="G4 steps")
plt.scatter(dataset['MCy'], dataset['MCz'], c = 'red', marker = '+', label="MC Truth")
plt.legend(loc='upper left')
plt.xlabel('y')
plt.ylabel('z')
plt.tight_layout()

plt.show()
