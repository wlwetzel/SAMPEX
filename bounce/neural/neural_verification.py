from sklearn.neural_network import MLPClassifier
import numpy as np
import plotly.express as px
import pandas as pd
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
from SAMP_Data import *
import plotly.graph_objects as go
from scipy import signal
from spacepy.time import Ticktock
from ast import literal_eval
from joblib import dump, load
from more_itertools import chunked
import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import copy
import numpy as np
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
import os


class verifyGui(Tk):
    """docstring for Window."""

    def __init__(self,  parent, year):
        Tk.__init__(self,parent)
        self.parent = parent
        self.path =  "/home/wyatt/Documents/SAMPEX/generated_Data/predictions_"+str(year)+".csv"
        self.accepted_path = "/home/wyatt/Documents/SAMPEX/generated_Data/accepted_predictions_"+str(year)+".csv"
        self.num_accepted=0
        self.num_rejected=0
        # try:
        #     os.remove(self.accepted_path)
        # except:
        #     pass
        self.initialize()

    def initialize(self):
        #load in data
        predictions = pd.read_csv(self.path,header=None,
                                  names=["Time","Counts","bounce"],usecols=[0,1,2])
        self.predictions = predictions.set_index('bounce',append=False)
        self.total_bounces = len(self.predictions.index.drop_duplicates())
        self.bounce_num = 0 #for keeping track of which bounce we're looking at

        #make figure and place button
        fig = Figure(figsize = (7, 7),
                     dpi = 100)
        plot1 = fig.add_subplot(111)
        self.line1, = plot1.plot(self.predictions.loc[self.bounce_num]['Time'],
                                 self.predictions.loc[self.bounce_num]['Counts'])

        self.canvas = FigureCanvasTkAgg(fig,
                                   master = self)
        self.canvas.draw()
        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().pack()
        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(self.canvas,
                                       self)
        toolbar.update()
        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().pack()

        #we need two buttons, which will store whether or not I think the
        #data contains a bounce
        acceptButton = Button(self,text="Accept",command = self.acceptBounce)
        rejectButton = Button(self,text="Reject",command = self.rejectBounce)
        flagButton   = Button(self,text="Flag",command = self.flagBounce)
        acceptButton.pack()
        rejectButton.pack()
        flagButton.pack()

        self.update()

    def refreshFigure(self):
        self.bounce_num+=1
        os.system('clear')
        print(f"{self.bounce_num} / {self.total_bounces}")
        false_pos = float(self.num_rejected) / (self.num_accepted + self.num_rejected)
        print(f"False Positive Rate: {false_pos * 100.0}")

        y = self.predictions.loc[self.bounce_num]['Counts'].to_numpy()
        x = np.arange(len(y))
        self.line1.set_data(x,y)
        ax = self.canvas.figure.axes[0]
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        self.canvas.draw()

    def acceptBounce(self):
        self.predictions.loc[self.bounce_num].to_csv(self.accepted_path,
                                                     mode='a',header=False)
        self.num_accepted+=1
        self.refreshFigure()

    def flagBounce(self):
        self.predictions.loc[self.bounce_num].to_csv("/home/wyatt/Documents/SAMPEX/generated_Data/flagged_Data.csv",
                                                     mode='a',header=False)
        self.refreshFigure()

    def rejectBounce(self):
        #pass on anything here for now
        self.num_rejected+=1
        self.refreshFigure()


window = verifyGui(None,96)
window.mainloop()
#
# predictions = pd.read_csv("/home/wyatt/Documents/SAMPEX/generated_Data/accepted_predictions_9.csv",
#                           header=None,names=["Bounce","Time","Counts"],usecols=[0,1,2])
# print(len(predictions["Bounce"].drop_duplicates()))
