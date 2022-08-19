import numpy as np

from scaleKMM import *
from util import *
from sklearn.preprocessing import MinMaxScaler

class Manager(object):

	def runEsnDensityRatio(self, traindata, trainBeta, testdata, gammab, splitSize, sampleSize, numSample, maxFeature):

		print('Estimating train bagging beta and ensemble test with split ' + str(splitSize) + ' and s = ' + str(numSample))
		beta, bagEnsSampled, bagEnsTime, bagEnsTimeTotal = scaleEnsKmm(traindata, testdata, gammab, sampleSize, numSample, maxFeature)

		beta_test, ensTrainTime, ensTrainTimeTotal = trainEnsKmm(traindata, testdata, gammab, sampleSize, maxFeature)

		return beta, beta_test




	#Starting beta computation for all three methods
	def runDensityRatio(self, count, traindata, trainBeta, testdata, maxFeature, splitSizeList, numSampleList):

		gammab = computeKernelWidth(traindata)

		print('Estimating full beta')

		fullbeta, fulltime = cenKmm(traindata, testdata, gammab, maxFeature)



		print('Estimating other beta ...')
		splitresult = {} #### <split : <num_sample : [nmseenste, timeenste, nmseenstr, timeenstr, nmsebag, timebag, nmsebagens, timebagens]>>
		rep = count
		beta = []

		for split in splitSizeList:

			sampleSize = int(len(traindata)/split) #m
			numSample = [computeNumSamples(traindata, 0.01, sampleSize)] #s
			if len(numSampleList) > 0:
				for s in numSampleList:
					numSample.append(s)

			numsampleresult = {}
			for s in numSample:

				for r in range(rep):

					beta_divide,beta_test = self.runEsnDensityRatio(traindata, trainBeta, testdata, gammab, split, sampleSize, s, maxFeature)
					beta_sum =np.sum(beta_divide)
					beta.append(beta_sum)
					print(f'beta_test:{beta_test}')
					print(f'beta:{beta_divide}')
					print(f'beta_sum:{beta_sum}')
					print(f'beta_list:{beta}')

		return fullbeta , beta


	#MAIN METHOD
	def start(self, count, trainSize, splitSize, numSampleList, datasetName, X, y):
		beta_2 = []
		full_2 = []
		for name in datasetName:

			# if name.endswith('.arff'):
			# 	data, label, maxFeature = getArffData(basedir + name, maxDatasetSize)
			# 	print(data)
			# 	print(maxFeature)
			# else:
			# 	data, label, maxFeature = getSparseData(basedir + name, maxDatasetSize)
			data, label, maxFeature = get_data(X,y)
			print(data)
			print(maxFeature)


			for c in range(count):
				traindata, trainBeta, testdata = generateTrain(data, trainSize)

				full, beta = self.runDensityRatio(count, traindata, trainBeta, testdata, maxFeature, splitSize, numSampleList)
				print(f'cenkmm_beta:{full}')
				beta_3 = np.sum(beta)
				beta_2.append(beta_3)
				#beta_2.append(beta)
				full_2.append(full)

			print(beta_2)
		return beta_2



#trian_old,trian_new,predict_old,predict_new
def main(X,y):

	count = 1
	trainSize = [5]
	#splitSize = [5,10,15,20] #k
	splitSize = [5]
	#numSampleList = [50,100,150,200] #s
	numSampleList = [50]  # s
	maxDatasetSize = 20

	train_data = X
	predict = y

	#Dataset File Names
	# datasetName = ['forestcover.arff', 'kdd.arff', 'pamap2.arff','powersupply.arff','sea.arff','syn002.arff', 'syn003.arff', 'mnist_100k_instances.data','news20_100k_instances.data']
	datasetName = ['forestcover.arff']
	#datasetName = ['covtype.csv']

	#datasetName = ['PRSA_Data_Aotizhongxin_20130301-20170228.csv']

	#Directory of dataset
	basedir = 'D:\\STUDY\\Deeplearning\\paper&code\\deadline\\sampling_kmm-master(veryfastKMM)\\sampling_kmm-master\\'
	#basedir = '/root/Documents/scale-kmm/dataset/'
	beta = []
	mgr = Manager()

	for t in trainSize:
		beta.append(mgr.start(count, t, splitSize, numSampleList, datasetName, train_data, predict))

	print(beta)
	sc = MinMaxScaler(feature_range=(0, 1))
	beta_final = sc.fit_transform(beta)
	print(f'beta_final:{beta_final}')

	return beta_final



if __name__ == '__main__':
    X = np.zeros([10,60,1])
    y = np.zeros([10])
    for i in range(10):
        for j in range(60):
            X[i,j,0] = i*100+j
        y[i] = i
    print(X, y)
    data, label, maxFeature = get_data(X, y)
    print(data)
    print(label)
    print(maxFeature)