import numpy as np
import pandas as pd

#candidates
years = [1994,1996,1997,1999,2001,2002,2003,2004]
total_bounces = 0
for year in years:
    path =  "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/candidate_"+str(year)+".csv"
    predictions = pd.read_csv(path,header=None,
                              names=["Time","Counts","bounce"],usecols=[0,1,2])
    predictions = predictions.set_index('bounce',append=False)

    total_bounces += len(predictions.index.drop_duplicates())
print(total_bounces)

#number used in stats
total_reviewed = 0
for year in years:
    counts_file = "/home/wyatt/Documents/SAMPEX/bounce/correlation/data/reviewed_"+str(year)+".csv"
    counts = pd.read_csv(counts_file,header=None,names=["Time","Counts","Peaks","Burst"],usecols=[0,1,2,3])
    counts['Time'] = pd.to_datetime(counts['Time'])
    peaks = counts[counts["Peaks"]==1][["Time","Burst"]]
    indices = list(set(peaks["Burst"].to_numpy()))
    total_reviewed+=len(indices)
print(total_reviewed)
