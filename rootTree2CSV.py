from ROOT import gSystem
from ROOT import *
import csv
import argparse
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None, help='specify an input file (ROOT ntuple)')
parser.add_argument('--output', type=str, default=None, help='specify an output file (csv file)')

my_args, _ = parser.parse_known_args()

filename = ""
filename_csv  = "out.csv"

if (my_args.input != None and os.path.isfile(my_args.input)):
    filename = my_args.input
else:
    print("Incorrect input file!! Exit the program!")
    sys.exit()

if (my_args.output != None):
    filename_csv = my_args.output

file = TFile(filename)
tree=file.Get("analysis")

with open(filename_csv, 'w') as csvfile:
    csv_writer=csv.writer(csvfile, delimiter=',')
    csv_writer.writerow(["trackNum", "MCx", "MCy" , "MCz"])

    for entry in tree:
        trackNum = tree.trackNum
        MCx = tree.MC_x
        MCy = tree.MC_y
        MCz = tree.MC_z

        csv_writer.writerow([trackNum, MCx, MCy, MCz])
