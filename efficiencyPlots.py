from ConformalTransform import conformalTransform, HoughTransform_phi, DoFullHT
from baseFunctions import read_csv_file, fit_stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

print(plt.style.available)
plt.style.use('seaborn-poster')
%matplotlib inline

energy = "100k"

DATA_PATH = './data/pgun/'
filename = "reco_"+energy+"MeV_theta90.csv"


df = read_csv_file(DATA_PATH, filename)

df.head()
max_track = 1  # df['trackNum'].max()
min_track = df['trackNum'].min()

R = []
a = []
b = []

for index_track in range(min_track, max_track):
    print("index_track: ", index_track)
    temp = df.loc[df['trackNum'] == index_track]
    radius_calc, a_calc, b_calc, phi = DoFullHT(temp['MCx'], temp['MCy'],
                                           mrange=[[0, np.pi], [-0.005, 0.005]],
                                           numpoints=500, binx=200, biny = 200, rangeConf=[(-0.003, 0.03), (-0.003, 0.03)], plotName = "")

    print("radius = ", radius_calc, ", phi = ", phi)
    R.append(radius_calc/1000.)
    a.append(a_calc/1000.)
    b.append(b_calc/1000.)

    figHits = plt.figure()
    ax = Axes3D(figHits)
    ax.scatter(temp['MCx'], temp['MCz'], temp['MCy'], c = 'red', marker = 'o', label="MC Truth")
    plt.legend(loc='upper left')
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_ylabel("y")
    plt.tight_layout()
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 20
    plt.show()

print(R)




# plt_R = plt.figure()
# mu_R, sigma_R, text_R = fit_stats(R)
# sns.distplot(R, kde=False, bins=100, fit=stats.norm)
# plt.xlabel('Radius [m]')
# plt.ylabel('Events')
# plt.legend([text_R])
# plt.tight_layout()
# plt.savefig("plots/radius_"+energy+"MeV.pdf")
#
# plt_a = plt.figure()
# mu_a, sigma_a, text_a = fit_stats(a)
# sns.distplot(a, kde=False, bins=100, fit=stats.norm)
# plt.xlabel('a [m]')
# plt.ylabel('Events')
# plt.legend([text_a])
# plt.tight_layout()
# plt.savefig("plots/a_"+energy+"MeV.pdf")
#
# plt_b = plt.figure()
# mu_b, sigma_b, text_b = fit_stats(b)
# sns.distplot(b, kde=False, bins=100, fit=stats.norm)
# plt.xlabel('b [m]')
# plt.ylabel('Events')
# plt.legend([text_b])
# plt.tight_layout()
# plt.savefig("plots/b_"+energy+"MeV.pdf")

# temp = df.loc[df['trackNum'] == 0]
# type(temp['MCx'])
# temp.head()
# radius_calc, a_calc, b_calc = DoFullHT(temp['MCx'], temp['MCy'],
#                                        mrange=[[0, np.pi], [-0.003, 0.003]],
#                                        numpoints=500, binx=200, biny = 200, rangeConf=[(-0.003, 0.03), (-0.003, 0.03)], plotName = "")




######## Simple example of clustering ########
# X = np.array([[1, 2], [1, 4], [1, 0],
#               [10, 2], [10, 4], [10, 0]])
#
# plt.scatter(X[:, 0], X[:, 1])
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# kmeans.labels_
# kmeans.predict([[0, 0], [12, 3]])
# kmeans.cluster_centers_
