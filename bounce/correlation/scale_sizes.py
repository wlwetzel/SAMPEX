import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
import SAMP_Data as sp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np

Re = 6371.0

bounce_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/stats.csv"
bounce_stats = pd.read_csv(bounce_file,
                           names = ["time_diff","percent_diff","periods",
                            "hemisphere","L","MLT","lat","lon","vx","vy","vz",
                            "timestamp","drift_vel","total_diff","alt"]
                            ,usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

bounce_stats["dist"] =pd.DataFrame(data={"dist":bounce_stats["total_diff"].to_numpy()*bounce_stats["drift_vel"].to_numpy() *Re})
# bounce_stats = bounce_stats[bounce_stats["L"]<4]
# bounce_stats = bounce_stats[bounce_stats["dist"]<75]

fig=px.histogram(bounce_stats["dist"])
fig.update_layout(title_text = "Scale Size Distribution, Observation",
                  xaxis_title_text="Scale (km)")

fig.update_layout(showlegend=False)

fig.write_image("/home/wyatt/Documents/SAMPEX/bounce_figures/important/scale_hist_local.png")
fig.show()
fig = go.Figure(data=
    go.Scatterpolar(
        r = bounce_stats["L"].to_numpy(),
        theta = (bounce_stats["MLT"]*360/24.0).to_numpy(),
        mode = 'markers',
        marker=dict(
            size=10,
            color=bounce_stats["dist"], #set color equal to a variable
            colorscale='Viridis', # one of plotly colorscales
            showscale=True
        )
    ))
fig.update_layout(
    title_text="L vs MLT",
    polar = dict(
      angularaxis = dict(
          tickmode = "array",
          tickvals = [int(mlt*360/24.0) for mlt in range(0,24,4)],
          ticktext = ["0","4","8","12","16","20"]
        )))
fig.update_traces(marker_colorbar=dict(title="Scale (km)"),
                  marker=dict(size=10),
                  text=bounce_stats["dist"].to_numpy())
fig.write_image("/home/wyatt/Documents/SAMPEX/bounce_figures/important/scale_dial_local.png")

fig.show()

"""
map back to equator lobs * 1/sqrt2 * (L /robs)**3/2
"""
prefactor = (2**.5 )* (bounce_stats["L"].to_numpy() / (1 + bounce_stats["alt"]/Re) )**3/2
bounce_stats["eq_dist"] = prefactor * bounce_stats["dist"]

bins =10** np.linspace(np.log10(bounce_stats["eq_dist"].min()),np.log10(bounce_stats["eq_dist"].max()),50)
hist,edges = np.histogram(bounce_stats["eq_dist"].to_numpy(),bins=list(bins))

bar_widths  = (bins[1:]-bins[:-1])
bar_centers = (bins[1:]+bins[:-1])/2
bar_height  = hist/bar_widths
fig = go.Figure(data=[go.Bar(
    x=bar_centers,
    y=bar_height,
    width=bar_widths
)])
fig.update_xaxes(type="log")
fig.update_layout(title_text="Eq. Azim. Scale Size, (Logarithmic, scaled by bin width)",
                  xaxis_title_text="Scale Size (km)")
fig.update_layout(showlegend=False)

fig.write_image("/home/wyatt/Documents/SAMPEX/bounce_figures/important/log_eq_scales.png")

fig.show()

fig=px.histogram(bounce_stats["eq_dist"])
fig.update_layout(title_text = "Scale Size Distribution, Equatorial",
                  xaxis_title_text="Scale (km)")
fig.write_image("/home/wyatt/Documents/SAMPEX/bounce_figures/important/scale_hist_eq.png")
fig.update_layout(showlegend=False)

fig.show()
fig = go.Figure(data=
    go.Scatterpolar(
        r = bounce_stats["L"].to_numpy(),
        theta = (bounce_stats["MLT"]*360/24.0).to_numpy(),
        mode = 'markers',
        marker=dict(
            size=10,
            color=bounce_stats["eq_dist"], #set color equal to a variable
            colorscale='Viridis', # one of plotly colorscales
            showscale=True
        )
    ))
fig.update_layout(
    title_text="L vs MLT",
    polar = dict(
      angularaxis = dict(
          tickmode = "array",
          tickvals = [int(mlt*360/24.0) for mlt in range(0,24,4)],
          ticktext = ["0","4","8","12","16","20"]
        )))
fig.update_traces(marker_colorbar=dict(title="Scale (km)"),
                  marker=dict(size=10),
                  text=bounce_stats["eq_dist"].to_numpy())

fig.write_image("/home/wyatt/Documents/SAMPEX/bounce_figures/important/scale_dial_eq.png")

fig.show()
