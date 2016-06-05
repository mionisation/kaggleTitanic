import csv as csv
import numpy as np

# 1 - Read CSV table
trainSetPath = '../0_Data/train.csv'
with open(trainSetPath, 'r') as csvFileHandle:
	csvFile = csv.reader(csvFileHandle)
	header = next(csvFile)
	data = []
	for row in csvFile:
		data.append(row)
data = np.array(data)

# 2 - Compute Death Rate
noPeople = np.size(data[0::, 1].astype(np.float))
noDeaths = np.sum(data[0::, 1].astype(np.float))
deathRate = noDeaths/noPeople
print("Death Rate is {}".format(deathRate))

# 3 - Women and Men separate Death Rate
womenVec = data[0::, 4] == 'female'
menVec = data[0::, 4] == 'male'

womenInfo = data[womenVec, 1].astype(np.float)
menInfo = data[menVec, 1].astype(np.float)

deathRateWomen = np.sum(womenInfo)/np.size(womenInfo)
deathRateMen = np.sum(menInfo)/np.size(menInfo)

print("Death rate of Women is {}".format(deathRateWomen))
print("Death rate of Men is {}".format(deathRateMen))

# 4 - Reading the test data and writing the gender model as a csv
testSetPath = '../0_Data/test.csv'
predictionPath = 'genderBasedModel.csv'
with open(testSetPath, 'r') as csvTestFileHandle, open(predictionPath, "w") as predictionFileHandle:
	testFile = csv.reader(csvTestFileHandle)
	header = next(testFile)
	predictionFile = csv.writer(predictionFileHandle)
	predictionFile.writerow(["PassengerID", "Survived"])
	for row in testFile:
		# write ID and prediction
		if row[3] == 'female':
			predictionFile.writerow([row[0], '1'])
		else:
			predictionFile.writerow([row[0], '0'])
	print("Gender based predictions written to {}".format(predictionPath))
