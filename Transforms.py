import numpy as np
from numpy import unravel_index
from sklearn.cluster import DBSCAN
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
from mpl_toolkits.mplot3d import Axes3D

class Transforms:

    def __init__(self, event):
        self._EventData = event
        self._X, self._Y, self._Z = self._EventData.data
        self._XYp = [self.conformalTransform(x, y) for (x, y) in zip(self._X,
                    self._Y) if self.conformalTransform(x, y)[2]]
        self._Xp, self._Yp, self._Rsquared = zip(*self._XYp)

    def conformalTransform(self, x, y):
        rhit_squared = (x**2 + y**2)
        if rhit_squared:
            xp = x / rhit_squared
            yp = y / rhit_squared
            return xp, yp, rhit_squared
        else:
            print("r2: ", rhit_squared)
            return 0, 0, 0

    def rho(self, rhit_squared, xp, yp, phi):
        r = (xp * np.cos(phi) + yp * np.sin(phi))
        return r

    def rho_phi(self, rhit_squared, xp, yp, numpoints):
        phis = np.linspace(0, 2*np.pi, numpoints)
        rhos = [self.rho(rhit_squared, xp, yp, phi) for phi in phis]
        return phis, rhos

    def HoughTransform_phi(self, numpoints, binx, biny, myrange, plotName=""):
        # myrange=[[0, np.pi], [min(ht_rho), max(ht_rho)]]
        ht_phi = []
        ht_rho = []
        for i in range(0, len(self._Xp)):
            phis, rhos = self.rho_phi(self._Rsquared[i], self._Xp[i], self._Yp[i], numpoints)
            ht_phi.extend(phis)
            ht_rho.extend(rhos)

        H, xedges, yedges = np.histogram2d(ht_phi, ht_rho, bins = (binx, biny))

        print("min rho: ", min(ht_rho))
        print("max rho: ", max(ht_rho))
        fig_HT_phi = plt.figure()
        h=plt.hist2d(ht_phi, ht_rho, bins=(binx, biny), cmap=plt.cm.jet, range=myrange)
        plt.colorbar(h[3])
        plt.xlabel(r'$\phi$ [rad]')
        plt.ylabel(r'$\rho$ [1/mm]')
        plt.tight_layout()
        if plotName:
            plt.savefig(plotName)
        else:
            plt.show()

        return H


    def cluster_test(self, H):
        am = H.argmax()
        r_idx = am % H.shape[1]
        c_idx = am // H.shape[1]

        trackpos = np.argwhere(H>112)

        print("size H: ", H.shape)
        # print("Ultimate max: ", r_idx, ", ", c_idx)
        print("Ultimate max: ", unravel_index(am, H.shape))
        print("Above threshold: ", trackpos.shape)
        print(H[trackpos[0][1]][trackpos[0][1]])
        print(trackpos[0][0])
        print(trackpos)

        # clustering = AffinityPropagation().fit(trackpos)

        # bandwidth = estimate_bandwidth(trackpos, quantile=0.3)
        # clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        clustering = DBSCAN(eps=1, min_samples=2).fit(trackpos)
        clustering.fit(trackpos)
        labels = clustering.labels_
        # clu_center = clustering.cluster_centers_
        print("labels: ", labels)
        # print("centers: ", clu_center)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print ("num clusters: ", n_clusters)
        print("H at centers:", [H[pos[0]][pos[1]] for pos in trackpos])


        # def DoFullHT(self, mrange, rangeConf, numpoints=500, binx=200, biny = 200, plotName=""):
        #     Xp, Yp, rhit_squared = zip(*XY_p)
        #     fig_conformalTransform = plt.figure()
        #     plt.scatter(Xp, Yp, c = 'DarkBlue', linestyle='-')
        #     plt.title("Conformal transform")
        #     plt.legend(loc='upper left')
        #     plt.xlabel("u")
        #     plt.ylabel("v")
        #     plt.xlim(rangeConf[0])
        #     plt.ylim(rangeConf[1])
        #     plt.tight_layout()
        #     # plt.show()
        #
        #     R_, theta_ = HoughTransform_phi(Rsquared=rhit_squared, Xp=Xp, Yp=Yp,
        #                                     numpoints=numpoints,
        #                                     binx=binx, biny=biny, myrange=mrange, plotName = plotName)
        #     radius_calc = 1./(2*R_)
        #     a_calc = np.cos(theta_)/(2*R_)
        #     b_calc = np.sin(theta_)/(2*R_)
        #
        #     return radius_calc, a_calc, b_calc, theta_
