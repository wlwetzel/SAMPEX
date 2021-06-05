# import matplotlib.pyplot as plt
# test = []
#
# def on_pick(event):
#     artist = event.artist
#     xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
#     x, y = artist.get_xdata(), artist.get_ydata()
#     ind = event.ind
#     print ('Artist picked:', event.artist)
#     print ('{} vertices picked'.format(len(ind)))
#     print ('Pick between vertices {} and {}'.format(min(ind),max(ind)+1))
#     print ('x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse))
#     print ('Data point:', x[ind[0]], y[ind[0]])
#     test.append(x[ind[0]])
#
# fig, ax = plt.subplots()
#
# tolerance = 10 # points
# ax.plot(range(10), 'ro-', picker=tolerance)
#
# fig.canvas.callbacks.connect('pick_event', on_pick)
#
# plt.show()
# print(test)

# import spacepy as _
# print(_.__path__)
# import pandas as pd
# df1 = pd.DataFrame(data = {"Time":[1,2,3] ,"Counts":[10,20,30]})
# df2 = pd.DataFrame(data = {"Time":[1.5,2.5,3.5] ,"Counts":[50,25,35]})
# df1['newlevel'] = 1
# df2['newlevel'] = 2
# df = pd.concat([df1, df2])
# df = df.set_index('newlevel', append=False)
# print(df)
# print(df.loc[1])
"""
"""
import pandas as pd

ind = [1,2,3,4,5]
dat = [x**2 for x in ind]
df = pd.DataFrame({"ind":ind,"dat":dat})
df["Burst"]=1
print(df)

quit()
"""
Make plots of flagged stuff
3 4 7 16 162 176 319
"""
import pandas as pd
import plotly.express as px
path = "/home/wyatt/Documents/SAMPEX/generated_Data/flagged_Data.csv"
df = pd.read_csv(path,names=['Burst','Time','Counts'])
df['Time'] = pd.to_datetime(df['Time'])

df = df.set_index(['Burst','Time'])

plotdf = df.loc[4]
fig = px.line(plotdf)
fig.update_layout(title_text = "Flagged Data")
fig.show()
