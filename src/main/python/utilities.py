#!/usr/bin/python

import re

#converts a comma separated string to a list
def arrayStringToList(arr):
	str = re.sub(r"[\n\t\s]", "", arr)
	str = str[1:]
	str = str[:len(str)-1]
	x = [float(i) for i in str.split(',')]
	return x