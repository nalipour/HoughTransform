import Hits
import Transforms
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None, help='specify an input file')
parser.add_argument('--background', type=str, default=None, help='specify an input file as background')
parser.add_argument('--output', type=str, default=None, help='specify an output path')
my_args, _ = parser.parse_known_args()


filename = ""
plotpath = "./"

if (my_args.input != None and os.path.isfile(my_args.input)):
    filename = my_args.input
else:
    print("Incorrect input file!! Exit the program!")
    sys.exit()


if my_args.output != None:
    plotpath = my_args.output

# DATA_PATH = "./data/pythia/"
# filename = "reco_simu_Zdd.csv"
#
# BCG_PATH = './data/'
# Filename_background = "reco_0.csv"


h = Hits.Hits(filename)
ev = Hits.Event(h) #, 11)

# h.drawAllEvents()
# Combine events (background, or several events)
# ev.combineEvents([Hits.Event(h_background)])
# ev.combineEvents([Hits.Event(h, 12), Hits.Event(h, 11)])


ev.drawEvent3D(plotName=plotpath+"3D_Zdd.pdf")
# ev.drawEventXY()#plotName=plotpath+"3tracks_XY.pdf")
# ev.drawEventXZ()
# ev.drawEventYZ()
d = ev.data
tr = Transforms.Transforms(ev)

H, xedges, yedges = tr.HoughTransform_phi(numpoints=200, binx=200,
                                          biny = 50, plotName = plotpath+"HT_Zdd_maxima.pdf")
tr.plotConformalTransform(plotpath+"CT_Zdd.pdf")
