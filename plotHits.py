import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

DATA_PATH = './data'
Filename = "mergedHits_2T.csv"
FilenameG4hits = "G4Hits_2T.csv"

csv_path = os.path.join(DATA_PATH, Filename)
csv_path2 = os.path.join(DATA_PATH, FilenameG4hits)
dataset = pd.read_csv(csv_path)
g4hits = pd.read_csv(csv_path2)

# Plot the raw hits
figHits = plt.figure()
ax = Axes3D(figHits)
#ax.scatter(g4hits['x'], g4hits['z'], g4hits['y'], c = 'blue', marker='x')
ax.scatter(dataset['MCx'], dataset['MCz'], dataset['MCy'], c = 'red', marker = 'o')



fig_xy = plt.figure()
# plt.scatter(g4hits['x'], g4hits['y'], c = 'blue', marker='x')
plt.scatter(dataset['MCx'], dataset['MCy'], c = 'red', marker = '+')


fig_xz = plt.figure()
# plt.scatter(g4hits['x'], g4hits['z'], c = 'blue', marker='x')
plt.scatter(dataset['MCx'], dataset['MCz'], c = 'red', marker = '+')


fig_yz = plt.figure()
# plt.scatter(g4hits['y'], g4hits['z'], c = 'blue', marker='x')
plt.scatter(dataset['MCy'], dataset['MCz'], c = 'red', marker = '+')

plt.show()
