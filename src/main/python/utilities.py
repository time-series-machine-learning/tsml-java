#!/usr/bin/python

import re

#converts a comma separated string to a list of ints
def array_string_to_list_int(arr):
	str = re.sub(r"[\n\t\s]", "", arr)
	str = str[1:]
	str = str[:len(str)-1]
	if str == '': return []
	x = [int(i) for i in str.split(',')]
	return x

#converts a comma separated string to a list of floats
def array_string_to_list_float(arr):
	str = re.sub(r"[\n\t\s]", "", arr)
	str = str[1:]
	str = str[:len(str)-1]
	if str == '': return []
	x = [float(i) for i in str.split(',')]
	return x

#converts a comma separated string to a list of strings
def array_string_to_list_string(arr):
	str = re.sub(r"[\n\t\s]", "", arr)
	str = str[1:]
	str = str[:len(str)-1]
	return str.split(',')

#converts a 2d semi-colona nd comma separated string to a list of floats
def deep_array_string_to_list_float(arr):
	x = [array_string_to_list_float(i) for i in arr.split(';')]
	return x