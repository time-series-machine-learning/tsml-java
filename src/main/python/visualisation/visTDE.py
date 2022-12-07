#!/usr/bin/python

#Author: Matthew Middlehurst

import sys
import numpy as np
from utilities import *
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "Calibri"

f = open(sys.argv[1] + 'vis' + sys.argv[2] + '.txt', 'r')

weight = float(f.readline())
l = f.readline().split()
rank = int(l[0])
num_classifiers = int(l[1])
l = f.readline().split()
word_length = int(l[0])
word_length_weight = float(l[1])
l = f.readline().split()
norm = bool(l[0])
norm_weight = float(l[1])
l = f.readline().split()
levels = int(l[0])
levels_weight = float(l[1])
l = f.readline().split()
igb = bool(l[0])
igb_weight = float(l[1])
l = f.readline().split()
window_length = int(l[0])
window_length_median = int(l[1])
weight_sum = float(f.readline())
class_counts = array_string_to_list_int(f.readline())
time_series = array_string_to_list_float(f.readline())
dft = array_string_to_list_float(f.readline())
word = f.readline()
breakpoints = deep_array_string_to_list_float(f.readline())

num_rows = 2+levels

## fig 1

fig = plt.figure()
gs = fig.add_gridspec(num_rows,4)

ax = fig.add_subplot(gs[0, :])
keys = array_string_to_list_string(f.readline())
counts = []
for i in range(int(sys.argv[3])):
	counts.append([n/class_counts[i] for n in array_string_to_list_float(f.readline())])
	ax.bar(keys, counts[i], bottom=counts_sum if i > 0 else 0, width=1, label='Class '+str(i), snap=False)
	counts_sum = [counts_sum[n] + counts[i][n] for n in range(len(counts_sum))] if i > 0 else counts[i]

l1_max = np.max(counts_sum)
ax.set_ylim(0, l1_max)
ax.set_title('Level 1')
ax.set_yticks([])
ax.set_xticks([])

plt.legend(bbox_to_anchor=(0, 1.52, 1, .102), loc='lower left', ncol=3, mode='expand', borderaxespad=0.)

if num_rows > 2:
	ax_q1 = fig.add_subplot(gs[2, :2])
	keys_q1 = array_string_to_list_string(f.readline())
	counts_q1 = []
	for i in range(int(sys.argv[3])):
		counts_q1.append([n/class_counts[i] for n in array_string_to_list_float(f.readline())])
		ax_q1.bar(keys_q1, counts_q1[i], bottom=counts_q1_sum if i > 0 else 0, width=1, label='Class '+str(i), snap=False)
		counts_q1_sum = [counts_q1_sum[n] + counts_q1[i][n] for n in range(len(counts_q1_sum))] if i > 0 else counts_q1[i]

	ax_q2 = fig.add_subplot(gs[2, 2:])
	keys_q2 = array_string_to_list_string(f.readline())
	counts_q2 = []
	for i in range(int(sys.argv[3])):
		counts_q2.append([n/class_counts[i] for n in array_string_to_list_float(f.readline())])
		ax_q2.bar(keys_q2, counts_q2[i], bottom=counts_q2_sum if i > 0 else 0, width=1, label='Class '+str(i), snap=False)
		counts_q2_sum = [counts_q2_sum[n] + counts_q2[i][n] for n in range(len(counts_q2_sum))] if i > 0 else counts_q2[i]

	l2_max = np.max(counts_q1_sum + counts_q2_sum)
	ax_q1.set_ylim(0, l2_max)
	ax_q1.set_yticks([])
	ax_q1.set_xticks([])
	ax_q1.set_title('Level 2 Q1')
	ax_q2.set_ylim(0, l2_max)
	ax_q2.set_yticks([])
	ax_q2.set_xticks([])
	ax_q2.set_title('Level 2 Q2')

	if num_rows > 4:
		ax_q3 = fig.add_subplot(gs[3, 0])
		keys_q3 = array_string_to_list_string(f.readline())
		counts_q3 = []
		for i in range(int(sys.argv[3])):
			counts_q3.append([n/class_counts[i] for n in array_string_to_list_float(f.readline())])
			ax_q3.bar(keys_q3, counts_q3[i], bottom=counts_q3_sum if i > 0 else 0, width=1, label='Class '+str(i), snap=False)
			counts_q3_sum = [counts_q3_sum[n] + counts_q3[i][n] for n in range(len(counts_q3_sum))] if i > 0 else counts_q3[i]

		ax_q4 = fig.add_subplot(gs[3, 1])
		keys_q4 = array_string_to_list_string(f.readline())
		counts_q4 = []
		for i in range(int(sys.argv[3])):
			counts_q4.append([n/class_counts[i] for n in array_string_to_list_float(f.readline())])
			ax_q4.bar(keys_q4, counts_q4[i], bottom=counts_q4_sum if i > 0 else 0, width=1, label='Class '+str(i), snap=False)
			counts_q4_sum = [counts_q4_sum[n] + counts_q4[i][n] for n in range(len(counts_q4_sum))] if i > 0 else counts_q4[i]

		ax_q5 = fig.add_subplot(gs[3, 2])
		keys_q5 = array_string_to_list_string(f.readline())
		counts_q5 = []
		for i in range(int(sys.argv[3])):
			counts_q5.append([n/class_counts[i] for n in array_string_to_list_float(f.readline())])
			ax_q5.bar(keys_q5, counts_q5[i], bottom=counts_q5_sum if i > 0 else 0, width=1, label='Class '+str(i), snap=False)
			counts_q5_sum = [counts_q5_sum[n] + counts_q5[i][n] for n in range(len(counts_q5_sum))] if i > 0 else counts_q5[i]

		ax_q6 = fig.add_subplot(gs[3, 3])
		keys_q6 = array_string_to_list_string(f.readline())
		counts_q6 = []
		for i in range(int(sys.argv[3])):
			counts_q6.append([n/class_counts[i] for n in array_string_to_list_float(f.readline())])
			ax_q6.bar(keys_q6, counts_q6[i], bottom=counts_q6_sum if i > 0 else 0, width=1, label='Class '+str(i), snap=False)
			counts_q6_sum = [counts_q6_sum[n] + counts_q6[i][n] for n in range(len(counts_q6_sum))] if i > 0 else counts_q6[i]

		l3_max = np.max(counts_q3_sum + counts_q4_sum + counts_q5_sum + counts_q6_sum)
		ax_q3.set_ylim(0, l3_max)
		ax_q3.set_yticks([])
		ax_q3.set_xticks([])
		ax_q3.set_title('Level 3 Q1')
		ax_q4.set_ylim(0, l3_max)
		ax_q4.set_yticks([])
		ax_q4.set_xticks([])
		ax_q4.set_title('Level 3 Q2')
		ax_q5.set_ylim(0, l3_max)
		ax_q5.set_yticks([])
		ax_q5.set_xticks([])
		ax_q5.set_title('Level 3 Q3')
		ax_q6.set_ylim(0, l3_max)
		ax_q6.set_yticks([])
		ax_q6.set_xticks([])
		ax_q6.set_title('Level 3 Q4')
	else:
		counts_q3 = []
		counts_q4 = []
		counts_q5 = []
		counts_q6 = []
		keys_q3 = []
		keys_q4 = []
		keys_q5 = []
		keys_q6 = []
		for i in range(int(sys.argv[3])):
			counts_q3.append([])
			counts_q4.append([])
			counts_q5.append([])
			counts_q6.append([])
else:
	counts_q1 = []
	counts_q2 = []
	counts_q3 = []
	counts_q4 = []
	counts_q5 = []
	counts_q6 = []
	keys_q1 = []
	keys_q2 = []
	keys_q3 = []
	keys_q4 = []
	keys_q5 = []
	keys_q6 = []
	for i in range(int(sys.argv[3])):
		counts_q1.append([])
		counts_q2.append([])
		counts_q3.append([])
		counts_q4.append([])
		counts_q5.append([])
		counts_q6.append([])

ax_bi = fig.add_subplot(gs[1, :])
keys_bi = array_string_to_list_string(f.readline())
counts_bi = []
for i in range(int(sys.argv[3])):
	counts_bi.append([n/class_counts[i] for n in array_string_to_list_float(f.readline())])
	ax_bi.bar(range(len(keys_bi)), counts_bi[i], bottom=counts_bi_sum if i > 0 else 0, width=1, label='Class '+str(i), snap=False)
	counts_bi_sum = [counts_bi_sum[n] + counts_bi[i][n] for n in range(len(counts_bi_sum))] if i > 0 else counts_bi[i]


if (len(counts_bi_sum) > 0):
	bi_max = np.max(counts_bi_sum)
	ax_bi.set_ylim(0, bi_max)
	ax_bi.set_yticks([])
	ax_bi.set_xticks([])
	ax_bi.set_title('Bigrams')

ax = fig.add_subplot(gs[num_rows - 1, :])
counts_conc = [counts[i] + counts_bi[i] + counts_q1[i] + counts_q2[i] + counts_q3[i] + counts_q4[i] + counts_q5[i] + counts_q6[i] for i in range(int(sys.argv[3]))]
keys_conc = keys + keys_bi + keys_q1 + keys_q2 + keys_q3 + keys_q4 + keys_q5 + keys_q6
keys_conc = range(len(counts_conc[0]))
for i in range(int(sys.argv[3])):
	ax.bar(keys_conc, counts_conc[i], bottom=counts_conc_sum if i > 0 else 0, width=1.0, label='Class '+str(i))
	counts_conc_sum = [counts_conc_sum[n] + counts_conc[i][n] for n in range(len(counts_conc_sum))] if i > 0 else counts_conc[i]

conc_max = np.max(counts_conc_sum)
ax.set_ylim(0, conc_max)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title('Concatenated and Weighted Features')
plt.subplots_adjust(wspace=0.1, hspace=0.5)

f.close()

plt.savefig(sys.argv[1] + 'vis' + sys.argv[2] + '.pdf')
plt.clf()

## fig 2

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(4,2)

ax = fig.add_subplot(gs[0, :])
ax.plot(range(1,len(time_series)+1), time_series)
ax.set_yticks([])
ax.set_title('Time Series')

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(range(1,window_length+1), time_series[:window_length])
ax2.set_yticks([])
ax2.set_title('1st Window')

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(range(1,len(dft)+1), dft)
ax3.set_yticks([])
ax3.set_title('Window DFT')

ax4 = fig.add_subplot(gs[2:, :])
ax4.set_yticks([])
ax4.set_title('Word Breakpoints')

max = -9999999999
min = 9999999999
for i in range(len(breakpoints)):
	for n in range(len(breakpoints[i])-1):
		if breakpoints[i][n] > max:
			max = breakpoints[i][n]
		if breakpoints[i][n] < min:
			min = breakpoints[i][n]

for i in range(len(dft)):
	if dft[i] > max:
		max = dft[i]
	if dft[i] < min:
		min = dft[i]

max = max + (max-min)/10
min = min - (max-min)/10
text_adjust = (max-min)/20
ch = ['a', 'b', 'c', 'd']

for i in range(1,word_length+1):
	if i < word_length:
		ax4.plot([i + .5, i + .5],[min,max],'r-')

	prev = []
	letters = []
	text_bottom = min
	for n in range(0,3):
		b = True
		for j in range(len(prev)):
			if breakpoints[i-1][n] <= prev[j]:
				b = False
				break

		if b:
			prev.append(breakpoints[i-1][n])
			letters.append(n)
			ax4.plot([i - .5, i + .5],[breakpoints[i-1][n],breakpoints[i-1][n]],'-r')

			ax4.text(i-.07,(breakpoints[i-1][n]+text_bottom-text_adjust)/2,ch[n])
			text_bottom = breakpoints[i-1][n]

	ax4.text(i-.07,(max+text_bottom-text_adjust)/2,ch[3])
	ax4.text(i-.07,min-4,word[i-1], fontsize=14, weight="bold")

ax4.plot(range(1,len(dft)+1), dft)

plt.savefig(sys.argv[1] + 'vis' + sys.argv[2] + '_2.pdf')
