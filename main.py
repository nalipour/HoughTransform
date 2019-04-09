import Hits
import Transforms
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

DATA_PATH = "./data/pythia/"
filename = "reco_simu_Zdd.csv"

BCG_PATH = './data/'
Filename_background = "reco_0.csv"


h = Hits.Hits(DATA_PATH, filename)
h_background = Hits.Hits(BCG_PATH, Filename_background)
# h.drawAllEvents()

# ev = Hits.Event(h) #, 11)
# # ev = Hits.Event(h_background)
# # Combine events (background, or several events)
# # ev.combineEvents([Hits.Event(h_background)])
# # ev.combineEvents([Hits.Event(h, 12), Hits.Event(h, 11)])
#
# plotpath = "./"
# ev.drawEvent3D(plotName=plotpath+"3D_Zdd.pdf")
# # ev.drawEventXY()#plotName=plotpath+"3tracks_XY.pdf")
# # ev.drawEventXZ()
# # ev.drawEventYZ()
# d = ev.data
# tr = Transforms.Transforms(ev)
#
# H, xedges, yedges = tr.HoughTransform_phi(numpoints=200, binx=200,
#                                           biny = 50, plotName = plotpath+"HT_Zdd_maxima.pdf")
# tr.plotConformalTransform(plotpath+"CT_Zdd.pdf")
