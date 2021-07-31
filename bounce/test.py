import numpy as np
import plotly.express as px
import pandas as pd
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
import SAMP_Data as sp
import plotly.graph_objects as go
from scipy import signal
from spacepy.time import Ticktock
from ast import literal_eval
from joblib import dump, load
from more_itertools import chunked
import tkinter
import spacepy.coordinates as spc
import spacepy.irbempy as irb
# from matplotlib.backends.backend_tkagg import (
#     FigureCanvasTkAgg, NavigationToolbar2Tk)
# # Implement the default Matplotlib key bindings.
# from matplotlib.backend_bases import key_press_handler
# from matplotlib.figure import Figure
# import copy
# import numpy as np
# from tkinter import *
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
# NavigationToolbar2Tk)
# import os
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import matplotlib.pyplot as plt
# # import matplotlib.pyplot as plt
# # test = []
# #
# # def on_pick(event):
# #     artist = event.artist
# #     xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
# #     x, y = artist.get_xdata(), artist.get_ydata()
# #     ind = event.ind
# #     print ('Artist picked:', event.artist)
# #     print ('{} vertices picked'.format(len(ind)))
# #     print ('Pick between vertices {} and {}'.format(min(ind),max(ind)+1))
# #     print ('x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse))
# #     print ('Data point:', x[ind[0]], y[ind[0]])
# #     test.append(x[ind[0]])
# #
# # fig, ax = plt.subplots()
# #
# # tolerance = 10 # points
# # ax.plot(range(10), 'ro-', picker=tolerance)
# #
# # fig.canvas.callbacks.connect('pick_event', on_pick)
# #
# # plt.show()
# # print(test)
#
# # import spacepy as _
# # print(_.__path__)
# # import pandas as pd
# # df1 = pd.DataFrame(data = {"Time":[1,2,3] ,"Counts":[10,20,30]})
# # df2 = pd.DataFrame(data = {"Time":[1.5,2.5,3.5] ,"Counts":[50,25,35]})
# # df1['newlevel'] = 1
# # df2['newlevel'] = 2
# # df = pd.concat([df1, df2])
# # df = df.set_index('newlevel', append=False)
# # print(df)
# # print(df.loc[1])
# """
# """
# # import pandas as pd
# #
# # ind = [1,2,3,4,5]
# # dat = [x**2 for x in ind]
# # df = pd.DataFrame({"ind":ind,"dat":dat})
# # df["Burst"]=1
# # print(df)
# #
# # quit()
# # """
# # Make plots of flagged stuff
# # 3 4 7 16 162 176 319
# # """
# # import pandas as pd
# # import plotly.express as px
# # path = "/home/wyatt/Documents/SAMPEX/generated_Data/flagged_Data.csv"
# # df = pd.read_csv(path,names=['Burst','Time','Counts'])
# # df['Time'] = pd.to_datetime(df['Time'])
# #
# # df = df.set_index(['Burst','Time'])
# #
# # plotdf = df.loc[4]
# # fig = px.line(plotdf)
# # fig.update_layout(title_text = "Flagged Data")
# # fig.show()
# Re = 6371
# L_list = []
# def pp(start, end, n):
#     start_u = start.value//10**9
#     end_u = end.value//10**9
#
#     return pd.DatetimeIndex((10**9*np.random.randint(start_u, end_u, n, dtype=np.int64)).view('M8[ns]'),tz='UTC')
# s = pd.to_datetime('1996-11-20')
# e = pd.to_datetime('2002-01-01')
# starts = pp(s,e,200)
# counter = 1
# # for start in starts:
# #     print(counter)
# #     counter+=1
# #     end = start+pd.Timedelta("5s")
# #     dataObj = sp.OrbitData(date=start)
# #     orbitInfo = dataObj.read_time_range(pd.to_datetime(start),pd.to_datetime(end),parameters=['GEI_X','GEI_Y','GEI_Z','L_Shell','GEO_Lat'])
# #
# #     X = (orbitInfo['GEI_X'].to_numpy() / Re)[0]
# #     Y = (orbitInfo['GEI_Y'].to_numpy() / Re)[0]
# #     Z = (orbitInfo['GElpI_Z'].to_numpy() / Re)[0]
# #     position  = np.array([X,Y,Z])
# #     ticks = Ticktock(start)
# #     coords = spc.Coords(position,'GEI','car')
# #     Lstar = irb.get_Lstar(ticks,coords,extMag='0')
# #     Lstar = abs(Lstar['Lm'][0])
# #     L_list.append(Lstar[0])
# # df = pd.DataFrame(data={"L":L_list})
# # df.to_csv("/home/wyatt/Documents/SAMPEX/bounce/correlation/ls.csv")
# df = pd.read_csv("/home/wyatt/Documents/SAMPEX/bounce/correlation/ls.csv")
# print(df)
# fig = px.histogram(df["L"][df["L"]<10],nbins=20)
# fig.show()

"""
making artificial bounces to use as kernels
"""
#three or four? idk
# burst = lambda t,amp,dist: amp * np.exp(-.5*(t-dist)**2 / (.075**2) )
# total_time = 5 #s
# samples = int(5/(20*10**-3))
# times = np.linspace(0,total_time,samples)
# bounce = None
# distance = .5 #s
# for n in range(1,5):
#     if bounce is None:
#         bounce = burst(times,1/n,n*distance)
#     else:
#         bounce += burst(times,1/n,n*distance)
# fig = px.line(x=times,y=bounce)
# fig.show()
a = 5

def foo(bar):
    return bar
print(a.foo())
