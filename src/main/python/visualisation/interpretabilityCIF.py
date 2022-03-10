#!/usr/bin/python

# Interpretability diagram generator for the Canonical Interval Forest classifier.
# Applicable to other interval forests.
# Inputs: figure save path, also used to load in tree nodes visited from text file
#         seed, used in save path
#         series id, used in save path
#         number of trees
#         series length
#         number of attributes used in the forests
#         class prediction
#
# Author: Matthew Middlehurst

import sys
import numpy as np
from statistics import median
from utilities import *
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "Calibri"

# Attribute names
names = ['DN_HistogramMode_5','DN_HistogramMode_10','SB_BinaryStats_mean_longstretch1','DN_OutlierInclude_p_001_mdrmd',
'DN_OutlierInclude_n_001_mdrmd','CO_f1ecac','CO_FirstMin_ac','SP_Summaries_welch_rect_area_5_1',
'SP_Summaries_welch_rect_centroid','FC_LocalSimple_mean3_stderr','CO_trev_1_num','CO_HistogramAMI_even_2_5',
'IN_AutoMutualInfoStats_40_gaussian_fmmi','MD_hrv_classic_pnn40','SB_BinaryStats_diff_longstretch0',
'SB_MotifThree_quantile_hh','FC_LocalSimple_mean1_tauresrat','CO_Embed2_Dist_tau_d_expfit_meandiff',
'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1','SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
'SB_TransitionMatrix_3ac_sumdiagcov','PD_PeriodicityWang_th0_01','Mean','Standard Deviation','Slope']

f = open(sys.argv[1] + 'pred' + sys.argv[2] + '-' + sys.argv[3] + '.txt', 'r')

# read in test series
f.readline()
series = array_string_to_list_float(f.readline())

# count how many times each time point is seen for each attribute over all tree nodes
count = np.zeros((int(sys.argv[6]),int(sys.argv[5])))
nodes = []
for i in range(int(sys.argv[4])):
	header = f.readline()
	# save nodes for trees which predicted the same class as the overall ensemble
	save_nodes = int(header.split()[7]) == int(sys.argv[7])
	for n in range(int(header.split()[3])-1):
		arr = array_string_to_list_string(f.readline())
		if save_nodes:
			nodes.append(arr)
		att = int(float(arr[0]))
		for g in range(int(float(arr[1])), int(float(arr[2])+1)):
			count[att][g] += 1
	f.readline()


# find top 3 attributes by max count for any time point
max = [max(i) for i in count]
top = sorted(range(len(max)), key=lambda i: max[i], reverse=True)[:3]

top_curves = [count[i] for i in top]
top_names = [names[i] for i in top]
top_times = [np.random.choice(np.flatnonzero(count[i] == count[i].max())) for i in top]

fig, axs = plt.subplots(3)
colours = ['orange','green','red']

# for each top attribute
for i in range(0,3):
    # find the average start and finish point for intervals containing the top time point for this attribute
	occurances = 0
	start = 0
	finish = 0
	for node in nodes:
		if float(node[0]) == top[i] and float(node[1]) <= top_times[i] and float(node[2]) >= top_times[i]:
			start += float(node[1])
			finish += float(node[2])
			occurances += 1
	start /= occurances
	finish /= occurances

	# find the closest node to the average start and end point
	closest = sys.maxsize
	for node in nodes:
		if float(node[0]) == top[i]:
			dist = abs(float(node[1]) - start) + abs(float(node[2]) - finish)

			if dist < closest or (dist == closest and float(node[2]) - float(node[1]) < float(closestNode[2]) - float(closestNode[1])):
				closest = dist
				closestNode = node

    # get interval for node
	interval = np.full(int(sys.argv[5]), np.nan)
	for n in range(int(float(closestNode[1])), int(float(closestNode[2]))):
		interval[n] = series[n]

	# get threshold for node
	threshold = float(closestNode[3])
	if float(closestNode[4]) == 1.0:
		sign = ">"
	else:
		sign = "<"

	# plot series with interval highlighted for this attribute
	axs[i].plot(series)
	axs[i].plot(interval, linewidth=3, color=colours[i], label=top_names[i] + "\n" + sign + "\n" + str(threshold))
	axs[i].set_yticks([])
	if i != 2:
		axs[i].set_xticks([])

	l = axs[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.savefig(sys.argv[1] + 'pred' + sys.argv[2] + '-' + sys.argv[3], bbox_inches='tight')