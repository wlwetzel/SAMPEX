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

class Window(Tk):
    """docstring for Window."""

    def __init__(self,  parent):
        Tk.__init__(self,parent)
        self.parent = parent
        self.path =  "/home/wyatt/Documents/SAMPEX/generated_Data/trimmed_predictions_94.csv"
        self.accepted_path = "/home/wyatt/Documents/SAMPEX/generated_Data/accepted_predictions_94.csv"
        self.initialize()

    def initialize(self):
        #load in data
        predictions = pd.read_csv(self.path,header=None,
                                  names=["Time","Counts","bounce"],usecols=[0,1,2])
        self.predictions = predictions.set_index('bounce',append=False)
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
        print(self.bounce_num)
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
        self.refreshFigure()

    def flagBounce(self):
        self.predictions.loc[self.bounce_num].to_csv("/home/wyatt/Documents/SAMPEX/generated_Data/flagged_Data.csv",
                                                     mode='a',header=False)
        self.refreshFigure()

    def rejectBounce(self):
        #pass on anything here for now
        self.refreshFigure()

#
# window = Window(None)
# window.mainloop()

predictions = pd.read_csv("/home/wyatt/Documents/SAMPEX/generated_Data/accepted_predictions_94.csv",
                          header=None,names=["Bounce","Time","Counts"],usecols=[0,1,2])
print(len(predictions["Bounce"].drop_duplicates()))
quit()
#trim down the predictions
path = "/home/wyatt/Documents/SAMPEX/generated_Data/predictions_94.csv"
trimmed_path = "/home/wyatt/Documents/SAMPEX/generated_Data/trimmed_predictions_94.csv"
predictions = pd.read_csv(path,header=None,names=["date","predictions"],usecols=[1,2])
predictions["predictions"] = predictions["predictions"].apply(literal_eval)
train_size = 151
bounce_num = 0
for date in predictions["date"]:
    print(date)
    try:
        obj  = HiltData(date=str(date))
        data = obj.read(None,None)
    except:
        print("No data available for this day")
        continue
    #loop over train_size length sections
    data_length = len(data.index)
    num_chunks = data_length/train_size
    df_dict = {n: data.iloc[n:n+train_size,:] for n in range(0,len(data.index),train_size)}
    row = predictions.loc[predictions['date'] == date]['predictions'].to_numpy()[0]
    counter=0
    for key,chunk in df_dict.items():
        if row[counter] !=0:
            print(row[counter])
            write = copy.deepcopy(chunk)
            write['bounce'] = bounce_num
            bounce_num +=1
            write.to_csv(trimmed_path,mode='a',header=False)
        counter+=1
