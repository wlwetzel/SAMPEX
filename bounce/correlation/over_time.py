import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot(df,fig,num):
    #L-MLT Dial Plot
    fig.add_trace(
        go.Scatterpolar(
            r = df["L"].to_numpy(),
            theta = (df["MLT"]*360/24.0).to_numpy(),
            mode = 'markers',
            marker=dict(
                size=10,
                color=df["period_comp"], #set color equal to a variable
                colorscale='Viridis', # one of plotly colorscales
                showscale=True
            )
        ),
        row=1,col=num
        )
    fig.update_layout(
        title_text="L vs MLT",
        polar = dict(
          angularaxis = dict(
              tickmode = "array",
              tickvals = [int(mlt*360/24.0) for mlt in range(0,24,4)],
              ticktext = ["0","4","8","12","16","20"]
            )))
    fig.update_traces(marker_colorbar=dict(title="Time Diff (Bounces)"),
                      text=df["period_comp"].to_numpy())



stats_file="/home/wyatt/Documents/SAMPEX/bounce/correlation/data/stats.csv"
df_master = pd.read_csv(stats_file,names = ["time_diff","percent_diff",
    "period_comp","hemisphere","L","MLT"],usecols=[1,2,3,4,5,6])
df_master = df_master[df_master["period_comp"]<1.75]
chunk_size = 200
data_dict = {n: df_master.iloc[n:n+chunk_size,:] for n in range(0,len(df_master.index),chunk_size)}
keys = {key for key in data_dict}
fig = make_subplots(rows=1,cols=4,
                    specs=[[{"type":"polar"},{"type":"polar"}
                           ,{"type":"polar"},{"type":"polar"}]])
id = 1
for key in keys:
    plot(data_dict[key],fig,id)
    id+=1
fig.show()
