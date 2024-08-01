import os
import csv
import sys

csv.field_size_limit(sys.maxsize)

num = 0;
cycles = 0;
instr = 0;
previous = "first"


while os.path.isdir(str(num)):
	os.remove(str(num) + '/trace_0.txt')
	cycles = 0;
	instr = 0;
	previous = "first"
	num = num+1




