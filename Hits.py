import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
from mpl_toolkits.mplot3d import Axes3D


# Looking into all the hits
class Hits:
    def __init__(self, path, filename):
        self._path = path
        self._filename = filename
        self._data = self.read_csv_file()
        self._MCx = self._data["MCx"]
        self._MCy = self._data["MCy"]
        self._MCz = self._data["MCz"]


    def read_csv_file(self):
        csv_path = os.path.join(self._path, self._filename)
        return pd.read_csv(csv_path)


    def returnEvent(self, event=0):
        if 'trackNum' in self._data:
            return self._data.loc[self._data['trackNum'] == event]
        else:
            return self._data


    def drawAllEvents(self):
        figHits = plt.figure()
        ax = Axes3D(figHits)
        ax.scatter(self._MCx, self._MCz, self._MCy, c = 'blue', marker = 'o', label="MC Truth")
        plt.legend(loc='upper left')
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_ylabel("y")
        plt.tight_layout()
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        ax.zaxis.labelpad = 20
        plt.show()


# Looking into each event
class Event:
    def __init__(self, hits, event=0):
        self._hits = hits
        self._data = hits.returnEvent(event)
        self.update()
        # self._MCx = self._data["MCx"]
        # self._MCy = self._data["MCy"]
        # self._MCz = self._data["MCz"]
        self._event = event


    @property
    def data(self):
        return self._MCx, self._MCy, self._MCz

    @property
    def data_df(self):
        return self._data

    @data.setter
    def data(self, event):
        self._event = event
        self._data = self._hits.returnEvent(event)
        self.update()
        # self._MCx = self._data["MCx"]
        # self._MCy = self._data["MCy"]
        # self._MCz = self._data["MCz"]


    def drawEvent3D(self):
        figHits = plt.figure()
        ax = Axes3D(figHits)
        ax.scatter(self._MCx, self._MCz, self._MCy, c = 'blue', marker = 'o', label="MC Truth")
        plt.legend(loc='upper left')
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_ylabel("y")
        plt.tight_layout()
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        ax.zaxis.labelpad = 20
        plt.show()

    def drawEventXY(self):
        fig_xy = plt.figure()
        plt.scatter(self._MCx, self._MCy, c = 'red', marker = '+', label="MC Truth")
        plt.legend(loc='upper left')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        plt.tight_layout()
        plt.show()

    def drawEventXZ(self):
        fig_xz = plt.figure()
        plt.scatter(self._MCx, self._MCz, c = 'red', marker = '+', label="MC Truth")
        plt.legend(loc='upper left')
        plt.xlabel('x [mm]')
        plt.ylabel('z [mm]')
        plt.tight_layout()
        plt.show()

    def drawEventYZ(self):
        fig_xz = plt.figure()
        plt.scatter(self._MCy, self._MCz, c = 'red', marker = '+', label="MC Truth")
        plt.legend(loc='upper left')
        plt.xlabel('y [mm]')
        plt.ylabel('z [mm]')
        plt.tight_layout()
        plt.show()

    def update(self):
        self._MCx = self._data["MCx"]
        self._MCy = self._data["MCy"]
        self._MCz = self._data["MCz"]

    def combineEvents(self, events):
        self._data = pd.concat([self._data]+[e.data_df for e in events], ignore_index=True)
        self.update()
