import corr_utils as corr
import pandas as pd
import plotly.express as px
# stats_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/stats.csv"
# df = pd.read_csv(stats_file,names = ["time_diff","percent_diff",
#                 "period_comp","hemisphere"],usecols=[1,2,3,4])
# df = df[df["period_comp"]<5]
# df["period_comp"] = df["period_comp"]-.2
# fig = px.histogram(df[['period_comp',"hemisphere"]],nbins=60,color="hemisphere")
# fig.update_layout(title_text = "Time Between Peaks Divided By Bounce Period",
#                     xaxis_title_text = "(Arb Units)")
# fig.show()

#find bounces
#[1994,1996,1997,
# years = [1998,1999,2000,2001,2002,2003,2004]
# for year in years:
#     blah = corr.corrSearch(year)
#     blah.search()

# year = 2004
# gu = corr.verifyGui(None,year)
# gu.mainloop()
# quit()

#identify peaks
# years = [1994,1996,1997,1998,1999,2000,
# years = [2001,2002,2003,2004]
# for year in years:
#     peak_obj = corr.peak_select(year)
#     peak_obj.select()
#make statistics
stat_obj = corr.stats_v2(stats_file="stats_60keV_30deg",energy=.06,mirr=30)
stat_obj.generate_stats(use_years=[1994,1996,1997,1999,2001,2002,2003,2004])
stat_obj.plot()
