import pandas as pd
from ast import literal_eval
import plotly.express as px
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
import SAMP_Data as sp
from scipy import signal
import plotly.graph_objects as go
import numpy as np
#
# stats_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/stats_60keV_30deg"
# stats = pd.read_csv(stats_file,names = ["time_diff","percent_diff",
#                 "period_comp","hemisphere","L"],usecols=[1,2,3,4,5])
# fig = px.scatter(x=stats["period_comp"],y=stats["L"])
# fig.update_layout(xaxis_title_text="Peak Diff in Bounce Periods",
#                   yaxis_title_text="L Shell")
# fig.show()
# print(px.data.wind())
# quit()
thetas_deg = [int(mlt*360/24.0) for mlt in range(0,24,4)]
strs = ["0","4","8","12","16","20"]

mlts = [0,4,8,12,16,20]
mlts = [mlt*360/24.0 for mlt in mlts]
vals = [1,2,3,4 ,5 ,6 ]
df = pd.DataFrame({"MLT":mlts,"val":vals})

fig = go.Figure(data=
    go.Scatterpolar(
        r = df["val"],
        theta = df["MLT"],
        mode = 'markers',
    ))
fig.update_layout(
    polar = dict(
      angularaxis = dict(
          tickmode = "array",
          tickvals = thetas_deg,
          ticktext = strs
        )
      )
)
fig.show()
