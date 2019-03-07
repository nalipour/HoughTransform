import Hits
import Transforms
import numpy as np
# %matplotlib inline

#energy = "100k"
energy = "2400"
DATA_PATH = './data/pgun/'
filename = "reco_"+energy+"MeV_theta90.csv"

BCG_PATH = './data/'
Filename_background = "reco_0.csv"


h = Hits.Hits(DATA_PATH, filename)
h_background = Hits.Hits(BCG_PATH, Filename_background)
# h.drawAllEvents()

ev = Hits.Event(h, 11)
# Combine events (background)
# ev.combineEvents([Hits.Event(h, 2)])#, Hits.Event(h_background)])

# print(ev._event)
# ev.data = 12
# ev._event
plotpath="/Users/nalipour/Documents/Fellow/Talks/WG11DetectorDesignMeeting/3_Mars7_2019/figures/"
plotname=plotpath+"track3D_2400MeV.pdf"
ev.drawEvent3D(plotName=plotname)
# ev.drawEventXY()
# ev.drawEventXZ()
# ev.drawEventYZ()
d = ev.data
tr = Transforms.Transforms(ev)

H = tr.HoughTransform_phi(myrange=[[0, np.pi], [-0.0004, 0.0004]],
         numpoints=200, binx=50, biny = 50, plotName = plotpath+"HT_2400MeV.pdf")


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
