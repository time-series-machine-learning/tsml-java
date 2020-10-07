#!/usr/bin/python

# Temporal importance curve diagram generator for interval forests.
# Applicable to other interval forests.
# Inputs: figure save path, also used to load in attribute/timepoint inforamtion gain from text file
#         seed, used in save path
#         number of attributes used in the forests
#         number of top attributes to plot
#
# Author: Matthew Middlehurst

import sys
import numpy as np
from utilities import *
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "Calibri"

f = open(sys.argv[1] + 'vis' + sys.argv[2] + '.txt', 'r')

# load attribute names and information gain data from txt file
curves = []
names = []
for i in range(int(sys.argv[3])):
	names.append(f.readline().strip())
	curves.append(array_string_to_list_float(f.readline()))

f.close()

# find num_atts attributes to display by max information gain for any time point
num_atts = int(sys.argv[4])
max = [max(i) for i in curves]
top = sorted(range(len(max)), key=lambda i: max[i], reverse=True)[:num_atts]

top_curves = [curves[i] for i in top]
top_names = [names[i] for i in top]

# plot curves with highest max and mean information gain for each time point
for i in range(0, num_atts):
	plt.plot(top_curves[i], label=top_names[i])
plt.plot(list(np.mean(curves, axis=0)), '--', linewidth=3, label='Mean Information Gain')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode='expand', borderaxespad=0.)
plt.xlabel('Time Point')
plt.ylabel('Information Gain')

plt.savefig(sys.argv[1] + 'vis' + sys.argv[2])