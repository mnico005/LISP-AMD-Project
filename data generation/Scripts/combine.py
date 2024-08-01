import pathlib
import os
import csv
import sys

csv.field_size_limit(sys.maxsize)
num = 0

locations = sorted(pathlib.Path('.').glob('**/sample.csv'))
with open("data.csv", "w") as outfile:
	fieldnames = ['stream', 'cpi']
	writer = csv.DictWriter(outfile, fieldnames=fieldnames)
	writer.writeheader()
	for i in locations:
		with open(i, newline='') as f:
			reader = csv.DictReader(f)
			num = 0
			for row in reader:
				writer.writerow({'stream':row['stream'], 'cpi':row['cpi']})
				print(i, num, "cpi:", row['cpi'])
				num = num +1
	#print(i)
