import csv
import sys

csv.field_size_limit(sys.maxsize)
num = 0
data = []
with open('data.csv', newline='') as f:
	reader = csv.DictReader(f)
	for row in reader:
		print(row['cpi'])

