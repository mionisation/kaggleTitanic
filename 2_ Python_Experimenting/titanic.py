import csv as csv
import numpy as np

# Read CSV table 
with open('../0_Data/train.csv', 'r') as csvFileHandle:
	csvFile = csv.reader(csvFileHandle)
	header = next(csvFile)
	data = []
	for row in csvFile:
		data.append(row)
data = np.array(data)
print(data)
