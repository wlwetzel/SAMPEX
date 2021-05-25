"""
testing with tkinter
"""
import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

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
        self.initialize()

    def initialize(self):
        fig = Figure(figsize = (5, 5),
                     dpi = 100)
        y = [i**2 for i in range(101)]
        plot1 = fig.add_subplot(111)
        self.line1, = plot1.plot(y)
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
        button = Button(self,text="Button",command = self.onButtonClick)
        button.pack()
        self.update()

    def refreshFigure(self,x,y):
        self.line1.set_data(x,y)
        ax = self.canvas.figure.axes[0]
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        self.canvas.draw()

    def onButtonClick(self):
        x = np.arange(101)
        y = np.random.rand(101)
        self.refreshFigure(x,y)

window = Window(None)
window.mainloop()
quit()
counter = 0
y = [i**2 for i in range(101)]
y2 = [i**.5 for i in range(101)]
ys = [y1,y2]
# plot function is created for
# plotting the graph in
# tkinter window
def plot():
    global counter

    # the figure that will contain the plot
    fig = Figure(figsize = (5, 5),
                 dpi = 100)

    # adding the subplot
    plot1 = fig.add_subplot(111)

    # plotting the graph
    plot1.plot(ys[counter])

    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig,
                               master = window)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()

    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas,
                                   window)
    toolbar.update()

    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()
    counter+=1

# the main Tkinter window
window = Tk()

# setting the title
window.title('Plotting in Tkinter')

# dimensions of the main window
window.geometry("500x500")

# button that displays the plot
plot_button = Button(master = window,
                     command = plot,
                     height = 2,
                     width = 10,
                     text = "Plot")

# place the button
# in main window
plot_button.pack()

# run the gui
window.mainloop()

quit()


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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
"""
checking obrien parameter
"""
def obrien(dat):
    # dat = pd.DataFrame(data={"Counts":dat})
    a_500 = dat.rolling(window=25,center=True).mean()
    n_100 = dat.rolling(window=5,center=True).mean()
    burst_param = (n_100-a_500)/np.sqrt(1+a_500)
    return burst_param
date = '1994138'
obj = HiltData(date=date)
data = obj.read(75600+3600-120,75600+3600-120+60)
burst = obrien(data)
data["Burst"]= burst
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=data["Counts"].index, y=data["Counts"].to_numpy()),
    secondary_y=False
)
fig.add_trace(
    go.Scatter(x=data["Burst"].index, y=data["Burst"].to_numpy()),
    secondary_y=True
)
fig.show()
quit()

"""
purpose is testing to see if I can use a neural network to predict if a given
segment of data has a bouncing microburst

ex. usage of sklearn
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)
print(clf.predict([[2., 2.], [-1., -2.]]))
"""

#make some sample time series
#let's make 4 with bounces, 2 that should be none
def make_test_data():
    time_num = 200
    gauss = lambda t,amp,sig , loc: amp * np.exp(-(t-loc)**2 / (sig**2))
    t = np.linspace(0,1,time_num)
    amps = [1,.5,.2]
    sigs = [.01,.008,.01]
    locs = [.2,.3,.4]
    ser1 = np.zeros(time_num)
    for i in range(3):
        ser1+=gauss(t,amps[i],sigs[i],locs[i])
    amps = [.6,.2,.1]
    sigs = [.02,.02,.01]
    locs = [.1,.27,.52]
    ser2 = np.zeros(time_num)
    for i in range(3):
        ser2+=gauss(t,amps[i],sigs[i],locs[i])
    amps = [.5,.4,.3,.2]
    sigs = [.01,.008,.01,.01]
    locs = [.46,.6,.86,.96]
    ser3 = np.zeros(time_num)
    for i in range(4):
        ser3+=gauss(t,amps[i],sigs[i],locs[i])
    amps = [1,.8,.6,.3]
    sigs = [.01,.008,.01,.01]
    locs = [.1,.3,.49,.7]
    ser4 = np.zeros(time_num)
    for i in range(4):
        ser4+=gauss(t,amps[i],sigs[i],locs[i])
    #should result in no bounces
    amps = [.2,.6,.8,1]
    sigs = [.01,.008,.01,.01]
    locs = [.1,.3,.49,.7]
    ser5 = np.zeros(time_num)
    for i in range(4):
        ser5+=gauss(t,amps[i],sigs[i],locs[i])

    amps = [.1,.8,.1,.3]
    sigs = [.11,.1,.1,.11]
    locs = [.1,.3,.49,.7]
    ser6 = np.zeros(time_num)
    for i in range(4):
        ser6+=gauss(t,amps[i],sigs[i],locs[i])

    amps = [.9,.7,.5]
    sigs = [.01,.015,.011]
    locs = [.2,.4,.6]
    test1 = np.zeros(time_num)
    for i in range(3):
        test1+=gauss(t,amps[i],sigs[i],locs[i])

    return np.stack((ser1,ser2,ser3,ser4,ser5,ser6)) , test1


data ,tests= make_test_data()
classes = [3 ,3 ,4 ,4 ,0 ,0]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,random_state=1)
clf.fit(data, classes)
print(clf.predict([tests]))

#this appears to have worked
