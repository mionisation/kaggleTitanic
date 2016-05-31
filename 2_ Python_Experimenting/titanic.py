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

# 5 - Pythonising the second submission

#Preprocess table to truncate high values
fare_ceiling = 40.0
data[data[0::, 9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling - 1.0
#Set up distinguishing features
fare_bracket_size = 10.0
number_of_brackets = fare_ceiling / fare_bracket_size
number_of_classes = len(np.unique(data[0::, 2]))

#Init survival table with all zeros for gender, number of classes, number of brackets
survival_table = np.zeros((2, number_of_classes, number_of_brackets))

for i in range(number_of_classes): #loop through each class
	for j in range(int(number_of_brackets)): #loop through price bins
		women_only_stats = data[												    #get ids of the passangers who:
							(data[0::,4]=='female')&								    #are female
							(data[0::,2].astype(np.float)== i+1)&				    #are in ith class
							(data[0::,9].astype(np.float)>=j*fare_bracket_size)&    #fare greater than this bin
							(data[0::,9].astype(np.float)< (j+1)*fare_bracket_size) #but less than next bin
							, 1]

		men_only_stats = data[														#get ids of the passangers who:
							(data[0::,4]!='female')&									#are female
							(data[0::,2].astype(np.float)== i+1)& 					#are in ith class
							(data[0::,9].astype(np.float)>=j*fare_bracket_size)& 	#fare greater than this bin
							(data[0::,9].astype(np.float)< (j+1)*fare_bracket_size) #but less than next bin
							, 1]

		survival_table[0, i, j] = np.mean(women_only_stats.astype(np.float))
		survival_table[1, i, j] = np.mean(men_only_stats.astype(np.float))
#Convert NaN to 0 (specific category combinations do not have any passengers)
survival_table[ survival_table != survival_table ] = 0.
# Probability of survival of passengers in gender, number of class, price bracket category
print(survival_table)
#If probability bigger than 0.5, passengers from that category should survive
survival_table[survival_table > 0.5] = 1
survival_table[survival_table <= 0.5] = 0
print(survival_table)
