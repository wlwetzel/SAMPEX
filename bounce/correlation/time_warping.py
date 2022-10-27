# Author: Romain Tavenard
# License: BSD 3 clause
from scipy.signal import find_peaks
from sklearn import preprocessing
from scipy.ndimage.filters import uniform_filter1d
from math import inf
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler
from tslearn.datasets import CachedDatasets
from tslearn.clustering import TimeSeriesKMeans
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import signal
import plotly.express as px
import sys
sys.path.append("/home/wyatt/Documents/SAMPEX")
import SAMP_Data as sp
import pandas as pd
import pyinform
energy=1
mc2 = .511 #elec rest mass
loss_cone = 3
beta = np.sqrt(energy/mc2)*np.sqrt(2 + energy/mc2) / (1 + energy/mc2)
Lstar = 7
period = .117 * Lstar * (1 - .4635 * np.sin(np.deg2rad(loss_cone))**.75) / beta
print(period)
quit()
year = 1994
day = 150
window = 10
start = pd.to_datetime("1994150221647", format="%Y%j%H%M%S",
                       utc=True)-pd.Timedelta("60s")
end = pd.to_datetime("1994150221651", format="%Y%j%H%M%S",
                     utc=True)+pd.Timedelta("80s")

obj = sp.HiltData(date=str(year)+str(day))
data = obj.read(None, None)[start:end]

data = data - data.rolling(window, min_periods=1).quantile(.1)
# data = data- data.min()
# data = data.to_numpy()
# data = data.flatten()
# data = data.reshape(-1,1)
#
# pt = preprocessing.QuantileTransformer(output_distribution="normal")
# proc = pt.fit_transform(data)
# fig = make_subplots(rows=2,cols=1)
# fig.add_trace(go.Scatter(x=[i for i in range(len(data.flatten()))],y=data.flatten()),row=1,col=1)
# fig.add_trace(go.Scatter(x=[i for i in range(len(proc.flatten()))],y=proc.flatten()),row=2,col=1)
# fig.show()
# quit()
data = data.to_numpy().flatten()
mean = np.mean(data)
std = np.std(data)
data = (data-mean)/std
# data = data - data.min()
data = data/data.max()
#### peak finding based algorithm

peaks,_ = find_peaks(data, height=None, threshold=None, distance=10, prominence=.1, width=None, wlen=None, rel_height=0.5, plateau_size=None)

trace = go.Scatter(x=list(range(len(data))),y=data)
fig = go.Figure(data=trace)
fig.add_trace(go.Scatter(x=peaks,y=data[peaks],mode='markers'))
fig.show()
quit()
# try rolling zscore
# r = data.rolling(window=150)
# m = r.mean().shift(1)
# s = r.std(ddof=0).shift(1)
# data = (data-m)/s
# data=data.to_numpy()
# try rolling normalization
# r = data.rolling(window=25)
# m = r.max()
# data = m/data

# peaks,_ = find_peaks(data,distance=4,prominence=.1)
# plt.plot(data)
# plt.plot(peaks, data[peaks], "x")
# plt.show()

kernel = data[3076:3182]
# peaks, _ = find_peaks(data, height=None, threshold=None, distance=None,
#                       prominence=.1, width=None, wlen=None, rel_height=0.5, plateau_size=None)

# slopes= np.zeros(len(data)-1)
# for n in range(len(peaks)-1):
#     slopes[peaks[n]]=(data[peaks[n+1]] - data[peaks[n]]) / (peaks[n+1] - peaks[n])


# diff = np.diff(data)
# dat_window = np.lib.stride_tricks.sliding_window_view(slopes, 75)
# slopes = np.mean(dat_window,axis=-1)
# fig = make_subplots(rows=2,cols=1)
# fig.add_trace(go.Scatter(x=list(range(len(slopes))),y=slopes),row=2,col=1)
# fig.add_trace(go.Scatter(x=peaks,y=data[peaks],mode='markers'),row=1,col=1)
# fig.add_trace(go.Scatter(x=list(range(len(data))),y=data),row=1,col=1)
# fig.show()
# quit()
def dtw(x, x_prime, q=2):
    R = np.zeros((len(x), len(x_prime)))
    for i in range(len(x)):
        for j in range(len(x_prime)):
            R[i, j] = abs(x[i] - x_prime[j]) ** q
            if i > 0 or j > 0:
                R[i, j] += min(
                    R[i-1, j] if i > 0 else inf,
                    R[i, j-1] if j > 0 else inf,
                    R[i-1, j-1] if (i > 0 and j > 0) else inf
                    # Note that these 3 terms cannot all beW
                    # inf if we have (i > 0 or j > 0)
                )
    return R ** 1./q
    # return R[-1, -1] ** (1. / q)


def DTWdistance(a, b):
    # z-normalize first
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    a_std = np.mean(a)
    b_std = np.mean(b)
    aa = (a-a_mean) / a_std
    bb = (b-b_mean) / b_std
    a_len = len(a)
    b_len = len(b)
    DTW = np.zeros((a_len+1, b_len+1))
    for i in range(a_len+1):
        for j in range(b_len+1):
            DTW[i, j] = inf
    DTW[0, 0] = 0
    # cost function will just be abs magnitude between points
    for i in range(1, a_len+1):
        for j in range(1, b_len+1):
            cost = abs(aa[i-1] - bb[j-1])**2
            DTW[i, j] = cost + min([DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1]])
    return DTW**.5


def DTWdistanceW(a, b, window):
    # z-normalize first
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    a_std = np.mean(a)
    b_std = np.mean(b)
    aa = (a-a_mean) / a_std
    bb = (b-b_mean) / b_std
    a_len = len(a)
    b_len = len(b)
    w = max([window, abs(a_len-b_len)])
    DTW = np.zeros((a_len+1, b_len+1))
    for i in range(a_len+1):
        for j in range(b_len+1):
            DTW[i, j] = inf
    DTW[0, 0] = 0
    for i in range(1, a_len+1):
        for j in range(max(1, i-1-w), min(b_len+1, i-1+w)):
            DTW[i, j] = 0
    # cost function will just be abs magnitude between points
    for i in range(1, a_len+1):
        for j in range(max(1, i-1-w), min(b_len+1, i-1+w)):
            cost = abs(aa[i-1] - bb[j-1])
            DTW[i, j] = cost + min([DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1]])
    return DTW


def path(D):
    N = D.shape[0]
    M = D.shape[1]
    # path will be backwards, start with [N-1,M-1]
    p = [[N-1, M-1]]
    while p[-1] != [0, 0]:
        n = p[-1][0]
        m = p[-1][1]

        if n == 0:
            p.append([0, m-1])
        elif m == 0:
            p.append([n-1, 0])
        else:
            # might be a better way to do this
            dic = {'a': D[n-1, m-1], 'b': D[n-1, m], 'c': D[n, m-1]}
            min_key = min(dic, key=dic.get)
            if min_key == 'a':
                p.append([n-1, m-1])
            elif min_key == 'b':
                p.append([n-1, m])
            elif min_key == 'c':
                p.append([n, m-1])
    p.reverse()
    return p


def slide(a, b):
    peaks, _ = find_peaks(data)

    slopes = np.zeros(len(data))
    for n in range(len(peaks)-1):
        slopes[peaks[n+1]] = (data[peaks[n+1]] -
                              data[peaks[n]]) / (peaks[n+1] - peaks[n])

    # a must be longer than b
    cost = [0]*(len(a)-2)
    def dist(x, y): return abs(x-y)**2
    # normalize b, and rolling normalize a in window the length of b
    b = b/np.max(b)
    for i in range(2, len(a)-len(b)):
        for j in range(len(b)-2):

            cost[i] += min(dist(a[i+j-2], b[j]),
                           dist(a[i+j-1], b[j]),
                           dist(a[i+j], b[j]),
                           dist(a[i+j+1], b[j]),
                           dist(a[i+j+2], b[j])
                           )
        # cost[i] += 10000*np.mean(slopes[i:i+len(a)])

    return np.array(cost)


def stretch(longer, short):
    # take short[j]->short[j*2]
    cost = np.zeros(len(longer)-len(short)*2)
    longer = longer/np.max(longer)
    short = short/np.max(short)
    for i in range(len(longer)-len(short)*2):
        #array[start:stop:step]

        cost[i] = min(np.mean(longer[i:i+len(short)] - short),
                      np.mean(longer[i:i+len(short)*2:2] - short))
    return cost


a = kernel.flatten()
b = data.flatten()
l1 = len(kernel)
l2 = len(data)
# l1 = 100
# l2 = 10000
t1 = np.linspace(0, 1, l1)
t2 = np.linspace(0, 10, l2)
r1 = [i for i in range(l1)]
r2 = [i for i in range(l2)]

# a = np.exp(-(t1-.3)**2 / .005) + np.random.normal(0,.05,l1)
# b = np.exp(-(t2-8)**2 / .005)  + np.random.normal(0,.05,l2) + np.exp(-(t2-1)**2 / .1)# + np.exp(-(t2-20)**2 / .5)
# b2 = np.exp(-(t2-5)**2 / .4)  + np.random.normal(0,.05,l2)#+ np.exp(-(t2-10)**2 / .8) + np.exp(-(t2-20)**2 / .5)


# DTW = DTWdistanceW(a,b,10)
# DTW = DTWdistance(b,a)
DTW = dtw(b, a)
# DTW2 = dtw(a,b2,3)
# DTW[DTW > 1e308] =0
p = path(DTW)
ns, ms = zip(*p)
# costs = np.diff(DTW[ns,ms])
# costs = DTW[ns,ms]
# costs = slide(b, a)
costs = slide(b, a)

# costs = uniform_filter1d(slide_plus(b,a),20)
# costs = costs[costs!=0]
# costs =uniform_filter1d(1.0/ costs,20)

# costs = costs[100:-100]
# print(dat_window.shape)
# print(costs.shape)
# costs =costs -  np.quantile(dat_window, .1,axis=-1)
# costs = ((cost_df - cost_df.rolling(window,min_periods=1).mean())**2 / (cost_df.rolling(window,min_periods=1).std())).to_numpy().flatten()
# costs = (costs/np.max(costs)).flatten()

# try to filter costs
costs = np.max(costs) - costs

sos = signal.butter(10,.1,output='sos')
costs = signal.sosfilt(sos,costs)

# corr = pyinform.active_info(costs,k=3,local=True).flatten()
fig = make_subplots(rows=2,cols=1)
fig.add_trace(go.Scatter(x=list(range(len(b))) , y=b ),row=1,col=1)
fig.add_trace(go.Scatter(x=list(range(len(costs))) , y=costs ),row=2,col=1)
fig.show()
quit()
peaks, _ = find_peaks(costs, height=None, threshold=None, distance=None,
                      prominence=None, width=None, wlen=None, rel_height=None, plateau_size=None)
# cost_df = pd.DataFrame(data=costs)
# means = cost_df.rolling(50,min_periods=1,center=True).quantile(.3).to_numpy().flatten()
# means = (costs-means)
# dat_window = np.lib.stride_tricks.sliding_window_view(means, 25)
# means = np.mean(dat_window,axis=-1)

fig = make_subplots(rows=3, cols=2,
                    row_heights=[.2, 0.2, 0.6],
                    column_widths=[.8, .2],
                    vertical_spacing=0.02,
                    horizontal_spacing=.02,
                    shared_yaxes=False,
                    shared_xaxes=True)
fig.add_trace(go.Heatmap(z=DTW.T), row=3, col=1)
# fig.add_trace(go.Scatter(x=ms,y=ns),row=3,col=1)
fig.add_trace(go.Scatter(x=list(range(len(costs))), y=costs), row=2, col=1)
fig.add_trace(go.Scatter(x=a.tolist(), y=r1), row=3, col=2)
fig.add_trace(go.Scatter(x=r2, y=b.tolist()), row=1, col=1)
# fig.add_trace(go.Scatter(x=peaks, y=costs[peaks], mode='markers',
#                          marker=dict(color='blue')), row=2, col=1)
# fig.add_trace(go.Scatter(x=list(range(len(means))) , y=means),row=2,col=1)
fig.show()

quit()
fig = px.line(data)
fig.show()
X_train = np.split(data[:-1], 50)
seed = 0
np.random.seed(seed)
# X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
# X_train = X_train[y_train < 4]  # Keep first 3 classes
# np.random.shuffle(X_train)
# Keep only 50 time series
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
clusters = 7
sz = X_train.shape[1]

# plt.figure()
sdtw_km = TimeSeriesKMeans(n_clusters=clusters,
                           metric="dtw",
                           # metric_params={"gamma": .01},
                           verbose=True,
                           random_state=seed)
y_pred = sdtw_km.fit_predict(X_train)

plt.figure()
for yi in range(clusters):
    plt.subplot(1, clusters, 1 + yi)

    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Soft-DTW $k$-means")

plt.tight_layout()
plt.show()

plt.figure()
for xx in X_train[y_pred == 0]:
    plt.plot(xx.ravel(), "k-", alpha=1)
    plt.show()
