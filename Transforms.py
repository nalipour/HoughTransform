import numpy as np
from sklearn.cluster import AffinityPropagation
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
            return 0, 0, 0

    def rho(self, rhit_squared, xp, yp, phi):
        r = (xp * np.cos(phi) + yp * np.sin(phi))
        return r

    def rho_phi(self, rhit_squared, xp, yp, numpoints):
        phis = np.linspace(0, 2*np.pi, numpoints)
        rhos = [self.rho(rhit_squared, xp, yp, phi) for phi in phis]
        return phis, rhos

    def HoughTransform_phi(self, numpoints, binx, biny, myrange, plotName):

        ht_phi = []
        ht_rho = []
        for i in range(0, len(self._Xp)):
            phis, rhos = self.rho_phi(self._Rsquared[i], self._Xp[i], self._Yp[i], numpoints)
            ht_phi.extend(phis)
            ht_rho.extend(rhos)

        H, xedges, yedges = np.histogram2d(ht_phi, ht_rho, bins = (binx, biny))
        am = H.argmax()
        r_idx = am % H.shape[1]
        c_idx = am // H.shape[1]

        trackpos = np.argwhere(H>112)

        print("size H: ", H.shape)
        print("Ultimate max: ", r_idx, ", ", c_idx)
        print("Above threshold: ", trackpos.shape)
        print(H[trackpos[0][1]][trackpos[0][1]])
        print(trackpos[0][0])
        print(trackpos)

        clustering = AffinityPropagation().fit(trackpos)
        print("labels: ", clustering.labels_)
        print("centers: ", clustering.cluster_centers_)
        # -------------------- #
        # kmeans = KMeans(n_clusters=1, random_state=0).fit(trackpos)
        # print("Labels: ", kmeans.labels_)
        # print("Cluster centers: ", kmeans.cluster_centers_)
        # -------------------- #
        # H[trackpos[0]])
        # print("x index = ", xedges[c_idx])
        # print("y index = ", yedges[r_idx])

        R_ = yedges[r_idx]
        theta_ = xedges[c_idx]

        # print("rho: ", R_)
        # print("phi: ", theta_, ", ", theta_ * 180 /np.pi, " deg")

        # print("Max R = ", max(ht_rho))
        fig_HT_phi = plt.figure()
        h=plt.hist2d(ht_phi, ht_rho, bins=(binx, biny), cmap=plt.cm.jet, range=myrange)
        plt.colorbar(h[3])
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'$\rho$')
        plt.tight_layout()
        plt.show()
        # plt.savefig('plots/'+plotName+'HT.pdf')
        radius_calc = 1./(2*R_)
        a_calc = np.cos(theta_)/(2*R_)
        b_calc = np.sin(theta_)/(2*R_)

        return R_, theta_

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
