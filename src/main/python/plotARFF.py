#!/usr/bin/python

#Script to plot a univariate time series ARFF with the class value as the final attribute.
#
#Author: Matthew Middlehurst

import sys
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "Calibri"

f = open(sys.argv[1], 'r')

line = f.readline()
while line.split()[0].strip().lower() != "@relation":
	line = f.readline()
	while line.strip() == "":
		line = f.readline()

num_atts = 0
line = f.readline().strip()
while line.lower() != "@data":
	if line != '':
		num_atts += 1
	line = f.readline().strip()

instances = []
classes = []
line = f.readline().strip()
while line != '':
	sp = line.split(',')
	instances.append(sp[:-1])
	classes.append(sp[-1])
	line = f.readline().strip()

class_set = list(set(classes))
x = range(num_atts-1)
for i in range(len(class_set)):
	l = [instances[n] for n in range(0, len(instances)) if classes[n] == class_set[i]]
	for n in range(len(l)):
		if n == 0:
			plt.plot(x, [float(g) for g in l[n]], 'C'+str(i), label=str(class_set[i]))
		else:
			plt.plot(x, [float(g) for g in l[n]], 'C'+str(i))

plt.legend()

plt.show()