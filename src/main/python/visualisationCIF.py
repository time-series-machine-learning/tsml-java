#!/usr/bin/python

import sys
from utilities import arrayStringToList

f = open(sys.argv[1] + "temporalImportanceCurves" + sys.argv[2] + ".txt", 'r')

curves = []
names = []
for i in range(int(sys.argv[3])):
	names.append(f.readline().strip())
	curves.append(arrayStringToList(f.readline()))
	
f.close()
	
print(names)