import os
import csv
import sys

csv.field_size_limit(sys.maxsize)

num = 0;
cycles = 0;
instr = 0;
previous = "first"


while os.path.isdir(str(num)):
	with open(str(num) + '/general.stat.out', 'r') as file:
		for line in file:
			for word in line.split():
				if previous == "INST_COUNT_TOT":
					instr = int(word)
				if previous == "CYC_COUNT_TOT":
					cycles = int(word)
				previous = word
	with open(str(num) + '/trace_0.txt', 'r', newline='') as file:
		#dictionary = { "stream" : file.read().replace(',', ''), "cpi": cycles/instr}
		with open("sample.csv", "a") as outfile:
			fieldnames = ['stream', 'cpi']
			writer = csv.DictWriter(outfile, fieldnames=fieldnames)
			if num == 0:
				writer.writeheader()
			writer.writerow({'stream':file.read().replace(',', '').replace('\n', '|'), 'cpi':cycles/instr})
			print("CPI",num,":", cycles/instr) 
    			#outfile.write("\n")
	
    			
	#print(num, "cpi:", cycles/instr)
	cycles = 0;
	instr = 0;
	previous = "first"
	num = num+1


