import corr_utils as corr
#find bounces
# year = 1994
# blah = corr.corrSearch(year)
# blah.search()
# gu = corr.verifyGui(None,year)
# gu.mainloop()
# quit()
#identify peaks
#years = [1994,1996,1997,1998,1999,
# years = [2000,2001,2002,2003,2004]
# for year in years:
#     peak_obj = corr.peak_select(year)
#     peak_obj.select()
#make statistics
stat_obj = corr.stats()
# stat_obj.generate_stats()
stat_obj.plot()
