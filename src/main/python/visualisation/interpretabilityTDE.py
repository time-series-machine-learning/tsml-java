#!/usr/bin/python

#Author: Matthew Middlehurst

import sys
import numpy as np
from utilities import *
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "Calibri"

f = open(sys.argv[1] + 'pred' + sys.argv[2] + '-' + sys.argv[3] + '.txt', 'r')

levels = int(sys.argv[4])
cls1 = str(sys.argv[5])
cls2 = str(sys.argv[6])

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(4,1)

ax = fig.add_subplot(gs[0, :])
time_series = array_string_to_list_float(f.readline())
ax.plot(time_series)
ax.set_title('Time Seires   Class: ' + cls1)
ax.set_yticks([])
ax.set_xticks([])

f.readline()
counts = array_string_to_list_int(f.readline())
if levels > 1:
	f.readline()
	counts_q1 = array_string_to_list_int(f.readline())
	f.readline()
	counts_q2 = array_string_to_list_int(f.readline())

	if levels > 2:
		f.readline()
		counts_q3 = array_string_to_list_int(f.readline())
		f.readline()
		counts_q4 = array_string_to_list_int(f.readline())
		f.readline()
		counts_q5 = array_string_to_list_int(f.readline())
		f.readline()
		counts_q6 = array_string_to_list_int(f.readline())
	else:
		counts_q3 = []
		counts_q4 = []
		counts_q5 = []
		counts_q6 = []
else:
	counts_q1 = []
	counts_q2 = []
	counts_q3 = []
	counts_q4 = []
	counts_q5 = []
	counts_q6 = []
f.readline()
counts_bi = array_string_to_list_int(f.readline())

time_series_hist = counts + counts_bi + counts_q1 + counts_q2 + counts_q3 + counts_q4 + counts_q5 + counts_q6

ax = fig.add_subplot(gs[2, :])
nearest_neighbour = array_string_to_list_float(f.readline())
ax.plot(nearest_neighbour)
ax.set_title('Nearest Neighbour   Class: ' + cls2)
ax.set_yticks([])
ax.set_xticks([])

f.readline()
counts = array_string_to_list_int(f.readline())
if levels > 1:
	f.readline()
	counts_q1 = array_string_to_list_int(f.readline())
	f.readline()
	counts_q2 = array_string_to_list_int(f.readline())

	if levels > 2:
		f.readline()
		counts_q3 = array_string_to_list_int(f.readline())
		f.readline()
		counts_q4 = array_string_to_list_int(f.readline())
		f.readline()
		counts_q5 = array_string_to_list_int(f.readline())
		f.readline()
		counts_q6 = array_string_to_list_int(f.readline())
	else:
		counts_q3 = []
		counts_q4 = []
		counts_q5 = []
		counts_q6 = []
else:
	counts_q1 = []
	counts_q2 = []
	counts_q3 = []
	counts_q4 = []
	counts_q5 = []
	counts_q6 = []
f.readline()
counts_bi = array_string_to_list_int(f.readline())

nearest_neighbour_hist = counts + counts_bi + counts_q1 + counts_q2 + counts_q3 + counts_q4 + counts_q5 + counts_q6

r = range(len(time_series_hist))

time_series_diff = [time_series_hist[i] - nearest_neighbour_hist[i] for i in r]
nearest_neighbour_diff = [-i if i < 0 else 0 for i in time_series_diff]
time_series_diff = [i if i > 0 else 0 for i in time_series_diff]
time_series_correct = [time_series_hist[i] - time_series_diff[i] for i in r]
nearest_neighbour_correct = [nearest_neighbour_hist[i] - nearest_neighbour_diff[i] for i in r]

ax = fig.add_subplot(gs[1, :])
ax.bar(r,time_series_correct,width=1.0, snap=False)
ax.bar(r,time_series_diff,width=1.0,bottom=time_series_correct, snap=False)
y_sum1 = [time_series_diff[i] + time_series_correct[i] for i in r]
ax.set_title('Time Seires Hiistogram')

ax2 = fig.add_subplot(gs[3, :])
ax2.bar(r,nearest_neighbour_correct,width=1.0, snap=False)
ax2.bar(r,nearest_neighbour_diff,width=1.0,bottom=nearest_neighbour_correct, snap=False)
y_sum2 = [nearest_neighbour_diff[i] + nearest_neighbour_correct[i] for i in r]
ax2.set_title('Nearest Neighbour Histogram')

y_max = np.max(y_sum1 + y_sum2)
ax.set_ylim(0, y_max)
ax.set_yticks([])
ax.set_xticks([])
ax2.set_ylim(0, y_max)
ax2.set_yticks([])
ax2.set_xticks([])

plt.savefig(sys.argv[1] + 'pred' + sys.argv[2] + '-' + sys.argv[3] + '.pdf', bbox_inches='tight')
