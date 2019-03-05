import Hits
import Transforms
import numpy as np
# %matplotlib inline

energy = "100k"

DATA_PATH = './data/pgun/'
filename = "reco_"+energy+"MeV_theta90.csv"


h = Hits.Hits(DATA_PATH, filename)
# h.drawAllEvents()

ev = Hits.Event(h, 10)
# print(ev._event)
# ev.data = 12
# ev._event
# ev.drawEvent3D()
# ev.drawEventXY()
# ev.drawEventXZ()
# ev.drawEventYZ()
d = ev.data
tr = Transforms.Transforms(ev)

H = tr.HoughTransform_phi(myrange=[[0, np.pi], [-0.003, 0.003]],
         numpoints=500, binx=200, biny = 200, plotName = "")


tr.cluster_test(H)
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# X = [1, 2, 3]
# Y = [4, 5, 6]
# Z = [7, 8, 9]
#
#
# [(x, y, z) for x, y, z in zip(X, Y, Z)]
