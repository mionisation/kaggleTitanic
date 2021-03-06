import csv as csv
import numpy as np

# Read Train CSV table
trainSetPath = '../0_Data/train.csv'
testSetPath = '../0_Data/test.csv'
with open(trainSetPath, 'r') as csvFileHandle:
	csvFile = csv.reader(csvFileHandle)
	header = next(csvFile)
	data = []
	for row in csvFile:
		data.append(row)
data = np.array(data)

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

#Read test file again and write results to results file
predictionPath = 'genderSocioEconomicModel.csv'
with open(testSetPath, 'r') as csvTestFileHandle, open(predictionPath, "w") as predictionFileHandle:
	testFile = csv.reader(csvTestFileHandle)
	header = next(testFile)
	predictionFile = csv.writer(predictionFileHandle)
	predictionFile.writerow(["PassengerID", "Survived"])
	for row in testFile:
		for j in range(int(number_of_brackets)):
			try: 
				row[8] = float(row[8])
			except:
				bin_fare = 3 - float(row[1])
				break
			#Categorize fare class according to paid price
			if row[8] > fare_ceiling:
				bin_fare = number_of_brackets - 1
				break
			if (row[8] >= j * fare_bracket_size) and (row[8] < (j+1) * fare_bracket_size):
				bin_fare = j
				break
		if row[3] == 'female':
			predictionFile.writerow([row[0], "%d" % int(survival_table[ 0, float(row[1]) - 1, bin_fare ])])
		else:
			predictionFile.writerow([row[0], "%d" % int(survival_table[ 1, float(row[1]) - 1, bin_fare])])
print("Gender and Price Classes based predictions written to {}".format(predictionPath))

