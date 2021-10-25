import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
import SAMP_Data as sp
import pandas as pd
import plotly.express as px
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np

year = 1994
day = 150
start = pd.to_datetime("1994150221647",format="%Y%j%H%M%S",utc=True)-pd.Timedelta("10s")
end = pd.to_datetime("1994150221651",format="%Y%j%H%M%S",utc=True)+pd.Timedelta("10s")

peaks_1 = [43.7,44.36,44.84,45.46]
peaks_2 = [46.78,47.12,47.36,47.74]
peaks_3 = [48.76,49.04,49.3,49.62,49.88,50.16]
peaks_4 = [58.06,58.56,59.12,59.7]
peaks = [peaks_1,peaks_2,peaks_3,peaks_4]
bounces = [np.mean(np.diff(peak)) for peak in peaks]
std = [np.std(np.diff(peak)) for peak in peaks]
def _transform_data(data):
    """
    data: pandas dataframe, 20ms SAMPEX count data
    return: data with rolling 10th percentile subtracted
    TESTING: *Subtracting min value off data chunks to try to get more
             similar vals for correlation
             *also trying some small boxcar smoothing
    """
    rolled = data.rolling(5,min_periods=1).mean()
    subtracted = data - data.rolling(10,min_periods=1).quantile(.1)

    return subtracted

def _correlate(data,kernel):
    """
    data: pd dataframe, 20ms count data
    kernel: bounce to compare data to
    """
    length = len(data.index)
    #divide by length of the data to normalize the correlation
    subtracted_data = _transform_data(data)
    subtracted_kernel = _transform_data(kernel)
    #zncc normalization
    data_mean = subtracted_data.mean().to_numpy()[0]
    data_std = subtracted_data.std().to_numpy()[0]
    kernel_mean = subtracted_kernel.mean().to_numpy()[0]
    kernel_std = subtracted_kernel.std().to_numpy()[0]
    correlation = signal.correlate(subtracted_data - data_mean,
                                   subtracted_kernel-kernel_mean,
                                   mode='same')/(length*kernel_std*data_std)
    return correlation.flatten()

def _load_artifical_kernel(distance):
    #distnace should be provided in seconds
    burst = lambda t,amp,dist: amp * np.exp(-.5*(t-dist)**2 / (.05**2) )
    total_time = 5 #s
    samples = int(5/(20*10**-3))
    times = np.linspace(0,total_time,samples)
    bounce = None
    for n in range(1,5):
        if bounce is None:
            bounce = burst(times,1/n,n*distance)
        else:
            bounce += burst(times,1/n,n*distance)
    return pd.DataFrame(bounce)


obj = sp.sampexStats(start)
l_val = obj.get_Lstar().to_numpy()[0]
bounce_period = obj.get_bounce_period()
bounce_periods = [i / bounce_period for i in bounces]
std_bounce = [i/bounce_period for i in std]

obj  = sp.HiltData(date=str(year)+str(day))
data = obj.read(None,None)
data = data[start:end]

anno_loc_0 = [pd.to_datetime("1994150221644",format="%Y%j%H%M%S",utc=True)
              ,2000,f"{bounce_periods[0]:.2f} +/- {std_bounce[0]:.2f} bounces"]
anno_loc_1 = [pd.to_datetime("1994150221647",format="%Y%j%H%M%S",utc=True)
              ,1900,f"{bounce_periods[1]:.2f} +/- {std_bounce[1]:.2f} bounces"]
anno_loc_2 = [pd.to_datetime("1994150221649",format="%Y%j%H%M%S",utc=True)
              ,1800,f"{bounce_periods[2]:.2f} +/- {std_bounce[2]:.2f} bounces"]
anno_loc_3 = [pd.to_datetime("1994150221658",format="%Y%j%H%M%S",utc=True)
              ,1200,f"{bounce_periods[3]:.2f} +/- {std_bounce[3]:.2f} bounces"]

anno_list = [anno_loc_0,anno_loc_1,anno_loc_2,anno_loc_3]
fig = px.line(data)
fig.update_layout(title_text = f"Bounce Period = {bounce_period:.2f}s, L = {l_val:.2f}")
for anno in anno_list:
    fig.add_annotation(x=anno[0],y=anno[1],text=anno[2],
                        font=dict(
                                family="Courier New, monospace",
                                size=16,
                                color="Black")
                        )
write_path = "/home/wyatt/Documents/SAMPEX/bounce_figures/"
fig.write_html(write_path + "case_study.html",include_plotlyjs="cdn")
pio.write_image(fig, write_path + 'case_study.eps',scale=1 ,width=1500, height=700)
# fig.show()

# kernel = _load_artifical_kernel(.3)
# corr = _correlate(data,kernel)
#
# fig = make_subplots(rows=2,cols=1)
#
# fig.add_trace(
#     go.Scatter(x=data.index,y=corr,name="Correlation"),row=2,col=1
# )
# fig.add_trace(
#     go.Scatter(x=data.index,y=data["Counts"].to_numpy(),name="SAMPEX Data"),row=1,col=1
# )
#
# fig.show()
