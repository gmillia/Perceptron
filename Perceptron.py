import numpy as np
import matplotlib.pyplot as plt

#Entry point into the program
def main():
	epochs = 30
	learning = 0.1
	batch = 300
	#load data into matrices
	pre_data = np.loadtxt(open("mnist_train.csv", "rb"), delimiter=",", skiprows=1)
	pret_data = np.loadtxt(open("mnist_test.csv", "rb"), delimiter=",", skiprows=1)
	t = pre_data[:,0:1]  #cut first column (targets) 
	tt = pret_data[:,0:1]  #cut first column (targets) from test data

	#get main (training data)
	data = getData(pre_data)
	targets = getTargets(t, data.shape[0])
	weights = getWeights(data.shape[1])

	#returns multiple things
	tup = train(epochs, data, weights, targets, learning, batch)

	weights = tup[1]
	trainCorrect = tup[0]
	trainEpochs = tup[2]
	trainWeights = tup[3]

	#get test data
	testData = getData(pret_data)
	test_rows = testData.shape[0]
	testTargets = getTargets(tt, test_rows)

	#gets correct % prediction for each epoch (list)
	testCorrect = test(epochs, testData, trainWeights, testTargets, batch)

	#prints accuracy graph for training and test data
	accuracyGraph(trainCorrect, batch, trainEpochs, epochs, testCorrect)

	#printf confusion matrix for the test data
	confMat(testData, testTargets, weights, epochs, batch, learning)

#helper function to create initial weights matrix
def getWeights(cols):
	weights = np.random.uniform(low=-0.5, high = 0.5, size=(cols, 10))
	return weights

#helper function that creates a onehot matrix of targets from 
#stripped first column (targets)
def getTargets(t, rows):
	targets = np.zeros(shape=(rows, 10), dtype = int)
	rN = 0
	for target in t:
		targets[rN, int(target)] = 1;
		rN+=1

	return targets

#helper function that strips data matrix from the first column (targets)
#and appends bias vector to the end, as well as optimizes data
def getData(pre_data):
	#remove first column from the input data (targets)
	data = np.delete(pre_data, 0, 1)

	#optimize data 
	data = data / 255

	#create bias vector and concat bias into data
	rows = np.size(data, 0)  #alt: data.shape[0]
	cols = np.size(data, 1)  #alt: data.shape[1]
	bias = np.full((1, rows), 1);
	data = np.concatenate((data, bias.T), axis = 1)

	return data

#helper function that tests input data on the latest weights
#and returns list of orrect predictions per epoch
def test(epochs, data, weightsList, targets, batch):
	correctList = []
	correct = 0;

	for epoch in range(0, epochs):
		correct = 0;

		output = np.dot(data, weightsList[epoch])
		output = np.where(output > 0, 1, 0)

		for i in range(0, data.shape[0]):
			exp = np.argmax(targets[i:i+1, :])
			out = np.argmax(output[i:i+1, :])

			if exp == out:
				correct+=1

		pct = (correct / data.shape[0])*100
		correctList.append(pct)

	return correctList

#helper function that trains the network
#returns list with percent of correct prediction, latest weights, 
#list of epochs, and list of weights computed at each epoch
def train(epochs, data, weights, targets, learning, batch):
	weightsList = []
	weightsList.append(weights)
	correctList = []
	epochList = []
	correct = 0
	incorrect = 0
	pct = 0

	for epoch in range(0, epochs):
		correct = 0
		print("Epoch:", (epoch+1), "/", epochs)
		for i in range(0, batch):
			print("Iteration:" ,i+1, "/", batch)
			activations = np.dot(data, weights)  #vector of sums: 60k x 10
			result = np.where(activations > 0, 1, 0)  #vector of 1s and 0s: 60lx10

			#result [60k x 10] - targets [60 x 10] = [60k x 10]
			#weights [785 x 10] 
			weights = weights - (learning * np.dot(data.T, result - targets))

			out = result[i:i+1,:]
			exp = targets[i:i+1,:]
			same = np.array_equal(out, exp)

			if same: 
				correct+=1
			else:
				incorrect+=1

		pct = (correct / batch)*100

		correctList.append(pct)
		weightsList.append(weights)
		epochList.append(epoch)

	return (correctList, weights, epochList, weightsList)

#helper function that graphs accuracy per epoch for both training
#and testing data
def accuracyGraph(correctList, batch, epochList,epochs, testCorrect):
	plt.title("Accuracy Graph")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.axis([0,epochs,0,100])

	plt.plot(epochList, correctList, color = "green", label="Training")
	plt.plot(epochList, testCorrect, color = "red", label="Test")
	plt.legend()
	plt.show()

#helper function that computes the confusion matrix for the testing
#data using the latest (most accurate) weights
def confMat(data, targets, weights, epochs, batch, learning):
	#create conf matrix
	size = targets.shape[1]
	conf = np.zeros(shape=(size+1, size+1), dtype = int)
	corr = 0

	for i in range(size):
		conf[0,i+1] = i
		conf[i+1,0] = i

	output = np.dot(data, weights)
	output = np.where(output > 0, 1, 0)

	for i in range(0, data.shape[0]):
		exp = np.argmax(targets[i:i+1, :])
		out = np.argmax(output[i:i+1, :])

		if np.array_equal(exp, out):
			corr+=1

		conf[exp+1, out+1] += 1

	print(conf)
	print("Accuracy:", (corr / data.shape[0])*100, "%")
	print("Leraning rate:", learning)
	print("Epochs:", epochs)
	print("Batch size:", batch)

if __name__ == "__main__":
    main()