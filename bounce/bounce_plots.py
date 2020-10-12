import pandas as pd
from ast import literal_eval
import plotly.express as px
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
from SAMP_Data import *

file = '/home/wyatt/Documents/SAMPEX/generated_Data/bouncing_93_v2.csv'
cols = ['date','num','groups']

bounces = pd.read_csv(file,names=cols,usecols=[1,2,3])
bounces['groups'] = bounces['groups'].apply(literal_eval)
bounces = bounces[bounces['num']!=0]
print(bounces['num'].sum())
for i in range(len(bounces.index)):
    date =str(bounces['date'].iloc[i])
    obj = HiltData(date=date)
    data = obj.read(None,None)
    data = data['Rate1']
    group = bounces['groups'].iloc[i]
    #we need to pick a range for the data, loop through groups
    for sub in group:
        lower = sub[0]-20
        upper = sub[-1]+20
        peaks_df = data.iloc[sub]
        current_data = data.iloc[lower:upper]
        fig = px.line(current_data)
        fig.add_scatter(x=peaks_df.index,y=peaks_df.to_numpy(),mode='markers',
                        name="Identified Peaks")
        fig.update_layout(title_text=date)
        fig.show()

    # input("Press enter to continue")

# grouped,prominences = peak_algo(x)
# peaks = list(itertools.chain.from_iterable(grouped))
# heights = x[peaks]-prominences
# fig = px.line(x)
# fig.add_scatter( x=peaks,y=x[peaks] ,mode='markers')
# for i in range(len(peaks)):
#     fig.add_shape(
#     # Line Vertical
#     dict(
#     type="line",
#     x0=peaks[i],
#     y0=heights[i],
#     x1=peaks[i],
#     y1=x[peaks[i]],
#     line=dict(
#     color="Black",
#     width=1
#     )
#     ))
# fig.show()
