import math, numpy, sklearn.metrics.pairwise as sk
import sys, random, time
from cvxopt import matrix, solvers

#FILE OPERATIONS


#I/O OPERATIONS
#Read input sparse file
#data is an array of dictionaries
def getSparseData(filename, size):

	data = []
	label = []
	labelnames = []
	maxFeature = 0

	with open(filename) as f:
		content = f.readlines()

	for c in content:
		c = c.strip()
		if not c: continue
		instance = {}
		items = c.split(' ')
		if items[0] not in labelnames:
			labelnames.append(items[0])
		label.append(float(labelnames.index(items[0])))

		for i in range(1, len(items)):
			item = items[i].split(':')
			nfeature = int(item[0]) - 1
			instance[nfeature] = float(item[1])

			if nfeature > maxFeature:
				maxFeature = nfeature

		data.append(instance)

		if len(data) == size:
			break

	return data, label, (maxFeature + 1)



#read arff file
def getArffData(filename, size):
	data = []
	label = []
	labelnames = []
	maxFeature = 0

	with open(filename) as f:
		content = f.readlines()

	dataFlag = False
	for c in content:

		if not dataFlag:
			if '@data' in c.lower():
				dataFlag = True
			continue

		c = c.strip()
		if not c: continue
		instance = {}
		items = c.split(',')
		for i in range(len(items)-1):
			instance[i] = float(items[i])
		data.append(instance)
		if items[-1] not in labelnames:
			labelnames.append(items[-1])
		label.append(float(labelnames.index(items[-1])))
		maxFeature = len(items)-1

		if len(data) == size:
			break

	return data, label, maxFeature


def get_data(X, y):
    data = []
    label = []
    for i in range(X.shape[0]):
        data_i = {ind: v for ind, v in zip(range(X.shape[1]), X[i, :, 0])}
        data.append(data_i)
        label.append(y[i])
    maxFeature = X.shape[1]

    return data, label, maxFeature



#I/O OPERATIONS
#Write Output to file
def writeFile(filename, data):
	if len(data) == 0:
		return

	with open(filename, 'w') as f:
		for i in data:
			f.write(str(i) + '\n')

############################################################################
############################################################################
############################################################################

#DENSITY ESTIMATION
#KMM solving the quadratic programming problem to get betas (weights) for each training instance
def kmm(Xtrain, Xtest, sigma):
	n_tr = len(Xtrain)
	n_te = len(Xtest)

	#calculate Kernel
	print('Computing kernel for training data ...')
	K_ns = sk.rbf_kernel(Xtrain, Xtrain, sigma)
	#make it symmetric
	K = 0.5*(K_ns + K_ns.transpose())

	#calculate kappa
	print('Computing kernel for kappa ...')
	kappa_r = sk.rbf_kernel(Xtrain, Xtest, sigma)
	ones = numpy.ones(shape=(n_te, 1))
	kappa = numpy.dot(kappa_r, ones)
	kappa = -(float(n_tr)/float(n_te)) * kappa

	#calculate eps
	eps = (math.sqrt(n_tr) - 1)/math.sqrt(n_tr)

	#constraints
	A0 = numpy.ones(shape=(1,n_tr))
	A1 = -numpy.ones(shape=(1,n_tr))
	A = numpy.vstack([A0, A1, -numpy.eye(n_tr), numpy.eye(n_tr)])
	b = numpy.array([[n_tr*(eps+1), n_tr*(eps-1)]])
	b = numpy.vstack([b.T, -numpy.zeros(shape=(n_tr,1)), numpy.ones(shape=(n_tr,1))*1000])

	print('Solving quadratic program for beta ...')
	P = matrix(K, tc='d')
	q = matrix(kappa, tc='d')
	G = matrix(A, tc='d')
	h = matrix(b, tc='d')

	start = time.time()
	beta = solvers.qp(P, q, G, h)

	end = time.time()
	print([i for i in beta['x']])
	return [i for i in beta['x']], (end - start)


############################################################################
############################################################################
############################################################################

#OTHERS

#Convert sparse data to array
def convertSparseToList(data, maxFeature):
	convData = []
	for d in data:
		instance = []
		for i in range(maxFeature):
			instance.append(0.0)

		for i in d:
			instance[i] = d[i]
		convData.append(instance)
	return convData


#Compute mean in sparse format
def __computeMean(data):
	mean = {}
	for d in data:
		for i in d:
			if i in mean:
				mean[i] += d[i]
			else:
				mean[i] = d[i]

	for i in mean:
		mean[i] /= len(data)
	return mean


#Compute distance
def __computeDistanceSq(d1, d2):
	dist = 0
	for i in d1:
		if i in d2:
			#when d1 and d2 have the same feature
			dist += ((d1[i] - d2[i]) ** 2)
		else:
			#feature in d1 only
			dist += (d1[i] ** 2)
	for i in d2:
		#feature in d2 only
		if i not in d1:
			dist += (d2[i] ** 2)
	return dist


#Compute standard deviation from mean of sparse data
def __computeSTD(data, mean):
	print('computing distance for STD')
	distList = []
	for d in data:
		dist = __computeDistanceSq(d, mean)

		distList.append(dist)

	print('Computing STD')
	return numpy.std(distList)


#Compute probability of sampling instance as training data
def __computeProb(instance, mean, std):
	dist = __computeDistanceSq(instance, mean)
	return math.exp(-1 * dist / std)


#Sample training data from input sparse data
def generateTrain(origdata, trainsize):
	train = []
	trainBeta = []
	data = list(origdata)

	print('Computing mean & std')
	mean = __computeMean(data)
	std = __computeSTD(data, mean)

	print('Generating training data')

	count = 0
	stdcount = 1
	origstd = std

	while len(train) < trainsize:
		d = random.randint(0, len(data)-1)
		p = __computeProb(data[d], mean, std)
		x = random.uniform(0,1)
		if x < p:
			train.append(data[d])
			trainBeta.append(1.0/p)
			del data[d]
			count = 0
		else:
			count += 1

		if count == trainsize:
			count = 0
			stdcount += 1
			std = origstd*stdcount

	return train, trainBeta, data


#Kernel width is the median of distances between instances of sparse data
def computeKernelWidth(data):
	dist = []
	for i in range(len(data)):
		for j in range(i+1, len(data)):
			s = __computeDistanceSq(data[i], data[j])
			dist.append(math.sqrt(s))
	return numpy.median(numpy.array(dist))


#compute NMSE
def computeNMSE(estBeta, origBeta):
	estBetaSum = sum(estBeta)
	origBetaSum = sum(origBeta)

	nmse = 0
	for i in range(len(estBeta)):
		nmse += ((estBeta[i]/estBetaSum) - (origBeta[i]/origBetaSum)) ** 2

	return nmse / len(estBeta)


#Get number of samples
def computeNumSamples(data, eps, sampleSize):
	print(sampleSize)
	print(math.log10(1-(1.0/len(data))))
	print(len(data))
	return int(math.ceil(math.log10(eps)/(sampleSize*math.log10(1-(1.0/len(data))))))