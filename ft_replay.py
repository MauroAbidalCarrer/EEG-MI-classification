import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from collections import deque


plt.ioff()
plt.show()


plt.ion()
#%matplotlib qt5
#matplotlib.use('qtagg')

##################################
sfreq = 500 # sampling frequency
visible = 2000 # time shown in plot (in samples) --> 4 seconds

# initialize deques
dy1 = deque(np.zeros(visible), visible)
dy2 = deque(np.zeros(visible), visible)
dx = deque(np.zeros(visible), visible)

# get interval of entire time frame
interval = np.linspace(0, eeg.shape[0], num=eeg.shape[0])
interval /= sfreq # from samples to seconds

# define channels to plot
ch1 = 'Fp2.'
ch2 = 'C3..'


##################################
# define figure size
fig = plt.figure(figsize=(12,12))

# define axis1, labels, and legend
ah1 = fig.add_subplot(211)
ah1.set_ylabel("Voltage [\u03BCV]", fontsize=14)
l1, = ah1.plot(dx, dy1, color='rosybrown', label=ch1)
ah1.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=0.5)

# define axis2, labels, and legend
ah2 = fig.add_subplot(212)
ah2.set_xlabel("Time [s]", fontsize=14, labelpad=10)
ah2.set_ylabel("Voltage [\u03BCV]", fontsize=14)
l2, = ah2.plot(dx, dy2, color='silver', label=ch2)
ah2.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=0.5)
##################################

start = 0
# simulate entire data
while start+visible <= eeg.shape[0]:

    # extend deques (both x and y axes)
    dy1.extend(eeg[ch1].iloc[start:start+visible])
    dy2.extend(eeg[ch2].iloc[start:start+visible])
    dx.extend(interval[start:start+visible])

    # update plot
    l1.set_ydata(dy1)
    l2.set_ydata(dy2)
    l1.set_xdata(dx)
    l2.set_xdata(dx)

    # get mean of deques
    y1_mean = np.mean(dy1)
    y2_mean = np.mean(dy2)

    # set x- and y-limits based on their mean
    ah1.set_ylim(-120+y1_mean, 200+y1_mean)
    interval_start = interval[start]
    max_index = interval.shape[0] - 1
    end_index = start+visible if start+visible < max_index else max_index
    interval_end = interval[end_index]
    ah1.set_xlim(interval_start, interval_end)
    ah2.set_ylim(-60+y2_mean, 100+y2_mean)
    ah2.set_xlim(interval[start], interval[end_index])

    # control speed of moving time-series
    start += 250

    fig.canvas.draw()
    fig.canvas.flush_events()