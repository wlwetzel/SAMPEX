import pandas as pd
from ast import literal_eval
import plotly.express as px
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
import SAMP_Data as sp
from scipy import signal
import numpy as np

stats_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/stats_60keV_30deg"
stats = pd.read_csv(stats_file,names = ["time_diff","percent_diff",
                "period_comp","hemisphere","L"],usecols=[1,2,3,4,5])
fig = px.scatter(x=stats["period_comp"],y=stats["L"])
fig.update_layout(xaxis_title_text="Peak Diff in Bounce Periods",
                  yaxis_title_text="L Shell")
fig.show()
