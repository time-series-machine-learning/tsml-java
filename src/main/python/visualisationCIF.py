#!/usr/bin/python

import sys
import numpy as np
from utilities import arrayStringToList
from matplotlib import pyplot as plt

f = open(sys.argv[1] + "temporalImportanceCurves" + sys.argv[2] + ".txt", 'r')

curves = []
names = []
for i in range(int(sys.argv[3])):
	names.append(f.readline().strip())
	curves.append(arrayStringToList(f.readline()))
	
f.close()

max = [max(i) for i in curves]
top = sorted(range(len(max)), key=lambda i: max[i], reverse=True)[:3]

top_curves = [curves[i] for i in top]
top_names = [names[i] for i in top] 
	
for i in range(0,3):
	plt.plot(top_curves[i], label=top_names[i])
plt.plot(list(np.mean(curves, axis=0)), '--', linewidth=3, label="Mean Information Gain")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel("Time Point")
plt.ylabel("Information Gain") 

plt.savefig(sys.argv[1] + "temporalImportanceCurves" + sys.argv[2])