import Hits
import Transforms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.get_configdir()
# print(plt.style.available)
# plt.style.use('seaborn-poster')
# %matplotlib inline

#energy = "100k"
energy = "2400"
DATA_PATH = './data/pgun/'
filename = "reco_"+energy+"MeV_theta90.csv"

# DATA_PATH = "./data/pythia/"
# filename = "reco_simu_Zdd.csv"

BCG_PATH = './data/'
Filename_background = "reco_0.csv"


h = Hits.Hits(DATA_PATH, filename)
h_background = Hits.Hits(BCG_PATH, Filename_background)
# h.drawAllEvents()

ev = Hits.Event(h)
# Combine events (background)
# ev.combineEvents([Hits.Event(h_background)])
ev.combineEvents([Hits.Event(h, 12), Hits.Event(h, 11)])

# print(ev._event)
# ev.data = 12
# ev._event
plotpath="/Users/nalipour/Documents/Fellow/Talks/FCCeeMDI/31_March_2019/figures/"
# plotname="track3D_2400MeV.pdf"
ev.drawEvent3D()#plotName=plotpath+"Zdd_3D.pdf")
ev.drawEventXY()#plotName=plotpath+"3tracks_XY.pdf")
ev.drawEventXZ()
ev.drawEventYZ()
d = ev.data
tr = Transforms.Transforms(ev)

H, xedges, yedges = tr.HoughTransform_phi(myrange=[[0, np.pi], [-0.0004, 0.0004]],
         numpoints=400, binx=200, biny = 200)#, plotName = plotpath+"HT_3Tracks.pdf")
tr.plotConformalTransform(plotpath)#+"CT_3Tracks.pdf")

# tr.cluster_test(H, xedges, yedges)
