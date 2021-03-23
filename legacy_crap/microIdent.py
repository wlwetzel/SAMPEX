import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
sys.path.append('/home/wyatt/pyModules')
import SAMPEXreb

start = pd.Timestamp('1997-03-01 00:00:00', tz='utc')
end = pd.Timestamp('1997-03-30 23:59:59', tz='utc')

data_iterator = SAMPEXreb.IteratorsLists(4, start, end)
SAMPEXreb.search_over_files(data_iterator,verify=True)
