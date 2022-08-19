from scaleKMM import *
from util import *
import logging
import math, numpy
import sys, random, time


class Manager(object):

	logger = None

	def __init__(self):
		self.__class__.logger = logging.getLogger(__name__)
		self.__class__.logger.setLevel(logging.INFO)

		sh = logging.StreamHandler()
		sh.setLevel(logging.INFO)
		self.__class__.logger.addHandler(sh)


	def runEsnDensityRatio(self, traindata, trainBeta, testdata, gammab, splitSize, sampleSize, numSample, maxFeature):

		print('Estimating test ensemble beta with split ' + str(splitSize))
		ensBetaTe, ensTestTime, ensTestTimeTotal = testEnsKmm(traindata, testdata, gammab, sampleSize, maxFeature)
		nmseEnsTe = computeNMSE(ensBetaTe, trainBeta)
		self.__class__.logger.info('Ensemble Test '+str(splitSize)+'  : Time = ' + str(ensTestTime) + '; NMSE = ' + str(nmseEnsTe) + '; Total Time = ' + str(ensTestTimeTotal))

		print('Estimating train ensemble beta with split ' + str(splitSize))
		ensBetaTr, ensTrainTime, ensTrainTimeTotal = trainEnsKmm(traindata, testdata, gammab, sampleSize, maxFeature)
		nmseEnsTr = computeNMSE(ensBetaTr, trainBeta)
		self.__class__.logger.info( 'Ensemble Train '+str(splitSize)+'  : Time = ' + str(ensTrainTime) + '; NMSE = ' + str(nmseEnsTr) + '; Total Time = ' + str(ensTrainTimeTotal))

		print('Estimating train bagging beta with split ' + str(splitSize) + ' and s = ' + str(numSample))
		bagBetaTr, bagTrSampled, bagTrainTime, bagTrainTimeTotal = scaleKmm(traindata, testdata, gammab, sampleSize, numSample, maxFeature)

		newTrainBeta = []
		for i in bagTrSampled:
			newTrainBeta.append(trainBeta[i])
		nmseBagTr = computeNMSE(bagBetaTr, newTrainBeta)
		self.__class__.logger.info( 'Bagging Train '+str(splitSize)+'-'+str(numSample)+' : Time = ' + str(bagTrainTime) + '; NMSE = ' + str(nmseBagTr) + '; Total Time = ' + str(bagTrainTimeTotal))


		print('Estimating train bagging beta and ensemble test with split ' + str(splitSize) + ' and s = ' + str(numSample))
		bagEnsTr, bagEnsSampled, bagEnsTime, bagEnsTimeTotal = scaleEnsKmm(traindata, testdata, gammab, sampleSize, numSample, maxFeature)

		newTrainBeta = []
		for i in bagEnsSampled:
			newTrainBeta.append(trainBeta[i])
		nmseBagEnsTr = computeNMSE(bagEnsTr, newTrainBeta)
		self.__class__.logger.info( 'Bagging Train ENS '+str(splitSize)+'-'+str(numSample)+' : Time = ' + str(bagEnsTime) + '; NMSE = ' + str(nmseBagEnsTr) + '; Total Time = ' + str(bagEnsTimeTotal))

		return nmseEnsTe, ensTestTime, nmseEnsTr, ensTrainTime, nmseBagTr, bagTrainTime, nmseBagEnsTr, bagEnsTime




	#Starting beta computation for all three methods
	def runDensityRatio(self, count, traindata, trainBeta, testdata, maxFeature, splitSizeList, numSampleList):

		self.__class__.logger.info('Train Length = ' + str(len(traindata)))
		self.__class__.logger.info( 'Test Length = ' + str(len(testdata)))
		self.__class__.logger.info( 'Num of features = ' + str(maxFeature))
		self.__class__.logger.info( 'Got training and test data.')

		gammab = computeKernelWidth(traindata)

		print('Estimating full beta')

		fullbeta, fulltime = cenKmm(traindata, testdata, gammab, maxFeature)
		fullnmse = computeNMSE(fullbeta, trainBeta)
		self.__class__.logger.info( 'Full  : Time = ' + str(fulltime) + '; NMSE = ' + str(fullnmse))


		print('Estimating other beta ...')
		splitresult = {} #### <split : <num_sample : [nmseenste, timeenste, nmseenstr, timeenstr, nmsebag, timebag, nmsebagens, timebagens]>>
		rep = count

		for split in splitSizeList:

			sampleSize = int(len(traindata)/split) #m
			numSample = [computeNumSamples(traindata, 0.01, sampleSize)] #s
			if len(numSampleList) > 0:
				for s in numSampleList:
					numSample.append(s)

			numsampleresult = {}
			for s in numSample:
				nmse_ens_te = time_ens_te = 0
				nmse_ens_tr = time_ens_tr = 0
				nmse_scale = time_scale = 0
				nmse_scale_ens = time_scale_ens = 0

				for r in range(rep):
					testensnmse, testenstime, trainensnmse, trainenstime, trainbagnmse, trainbagtime, bagensnmse, bagenstime = self.runEsnDensityRatio(traindata, trainBeta, testdata, gammab, split, sampleSize, s, maxFeature)
					nmse_ens_te += testensnmse
					time_ens_te += testenstime
					nmse_ens_tr += trainensnmse
					time_ens_tr += trainenstime
					nmse_scale += trainbagnmse
					time_scale += trainbagtime
					nmse_scale_ens += bagensnmse
					time_scale_ens += bagenstime

				numsampleresult[s] = []
				numsampleresult[s].append(nmse_ens_te/rep)
				numsampleresult[s].append(time_ens_te/rep)
				numsampleresult[s].append(nmse_ens_tr/rep)
				numsampleresult[s].append(time_ens_tr/rep)
				numsampleresult[s].append(nmse_scale/rep)
				numsampleresult[s].append(time_scale/rep)
				numsampleresult[s].append(nmse_scale_ens/rep)
				numsampleresult[s].append(time_scale_ens/rep)

			splitresult[split] = numsampleresult


		return fullnmse, fulltime, splitresult


	#MAIN METHOD
	def start(self, count, trainSize, splitSize, numSampleList, maxDatasetSize, datasetName, basedir):

		handler = logging.FileHandler('scalekmm_'+str(trainSize)+'.log')
		handler.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		handler.setFormatter(formatter)
		self.__class__.logger.addHandler(handler)

		resultNMSE = {}
		resultTime = {}

		for name in datasetName:
			self.__class__.logger.info( 'Starting '+name)
			if name.endswith('.arff'):
				data, label, maxFeature = getArffData(basedir + name, maxDatasetSize)
			else:
				data, label, maxFeature = getSparseData(basedir + name, maxDatasetSize)
			self.__class__.logger.info('Read data '+ name)

			fullNMSE = 0
			teEnsNMSE = {}
			trEnsNMSE = {}
			trBagNMSE = {}
			bagEnsNMSE = {}

			fullTime = 0
			teEnsTime = {}
			trEnsTime = {}
			trBagTime = {}
			bagEnsTime = {}



			for c in range(count):
				traindata, trainBeta, testdata = generateTrain(data, trainSize)
				fn, ft, otherRes = self.runDensityRatio(count, traindata, trainBeta, testdata, maxFeature, splitSize, numSampleList)

				fullNMSE += fn
				fullTime += ft

				for k in otherRes:
					#Test ENS NMSE
					if k in teEnsNMSE:
						for s in teEnsNMSE[k]:
							teEnsNMSE[k][s] += otherRes[k][s][0]
					else:
						teEnsNMSE[k] = {}
						for s in otherRes[k]:
							teEnsNMSE[k][s] = otherRes[k][s][0]
					#Test ENS Time
					if k in teEnsTime:
						for s in teEnsTime[k]:
							teEnsTime[k][s] += otherRes[k][s][1]
					else:
						teEnsTime[k] = {}
						for s in otherRes[k]:
							teEnsTime[k][s] = otherRes[k][s][1]

					#Train ENS NMSE
					if k in trEnsNMSE:
						for s in trEnsNMSE[k]:
							trEnsNMSE[k][s] += otherRes[k][s][2]
					else:
						trEnsNMSE[k] = {}
						for s in otherRes[k]:
							trEnsNMSE[k][s] = otherRes[k][s][2]
					#Train ENS Time
					if k in trEnsTime:
						for s in trEnsTime[k]:
							trEnsTime[k][s] += otherRes[k][s][3]
					else:
						trEnsTime[k] = {}
						for s in otherRes[k]:
							trEnsTime[k][s] = otherRes[k][s][3]

					#Bag NMSE
					if k in trBagNMSE:
						for s in trBagNMSE[k]:
							trBagNMSE[k][s] += otherRes[k][s][4]
					else:
						trBagNMSE[k] = {}
						for s in otherRes[k]:
							trBagNMSE[k][s] = otherRes[k][s][4]
					#Bag Time
					if k in trBagTime:
						for s in trBagTime[k]:
							trBagTime[k][s] += otherRes[k][s][5]
					else:
						trBagTime[k] = {}
						for s in otherRes[k]:
							trBagTime[k][s] = otherRes[k][s][5]

					#Bag ENS NMSE
					if k in bagEnsNMSE:
						for s in bagEnsNMSE[k]:
							bagEnsNMSE[k][s] += otherRes[k][s][6]
					else:
						bagEnsNMSE[k] = {}
						for s in otherRes[k]:
							bagEnsNMSE[k][s] = otherRes[k][s][6]
					#Bag Time
					if k in bagEnsTime:
						for s in bagEnsTime[k]:
							bagEnsTime[k][s] += otherRes[k][s][7]
					else:
						bagEnsTime[k] = {}
						for s in otherRes[k]:
							bagEnsTime[k][s] = otherRes[k][s][7]

			# logger.info( '----------------'+name+'------------------------')
			# logger.info( 'Full : Time = ' + str(float(fullTime)/count) + '; NMSE = ' + str(float(fullNMSE)/count))
			# logger.info( 'TestENS : Time = ' + str(float(teEnsTime)/count) + '; NMSE = ' + str(float(teEnsNMSE)/count))
			# logger.info( 'TrainENS : Time = ' + str(float(trEnsTime)/count) + '; NMSE1 = ' + str(float(trEnsNMSE1)/count) + '; NMSE2 = ' + str(float(trEnsNMSE2)/count))
			# logger.info( '----------------------------------------------------')

			resultNMSE[name] = [float(fullNMSE)/count, teEnsNMSE, trEnsNMSE, trBagNMSE, bagEnsNMSE]
			resultTime[name] = [float(fullTime)/count, teEnsTime, trEnsTime, trBagTime, bagEnsTime]

		self.__class__.logger.info( '\n\n--------F I N A L------------')
		for name in resultNMSE:
			self.__class__.logger.info(name)
			self.__class__.logger.info('Full : Time = ' + str(resultTime[name][0]) + '; NMSE = ' + str(resultNMSE[name][0]))
			for k in resultTime[name][1]:
				for s in resultTime[name][1][k]:
					self.__class__.logger.info('TestENS '+str(k)+' - '+str(s)+' : Time = ' + str(float(resultTime[name][1][k][s])/count) + '; NMSE = ' + str(float(resultNMSE[name][1][k][s])/count))
					self.__class__.logger.info('TrainENS '+str(k)+' - '+str(s)+' : Time = ' + str(float(resultTime[name][2][k][s])/count) + '; NMSE = ' + str(float(resultNMSE[name][2][k][s])/count))
					self.__class__.logger.info('TrainBag '+str(k)+' - '+str(s)+' : Time = ' + str(float(resultTime[name][3][k][s])/count) + '; NMSE = ' + str(float(resultNMSE[name][3][k][s])/count))
					self.__class__.logger.info('BagENS '+str(k)+' - '+str(s)+' : Time = ' + str(float(resultTime[name][4][k][s])/count) + '; NMSE = ' + str(float(resultNMSE[name][4][k][s])/count))
			self.__class__.logger.info('------------------------')

		self.__class__.logger.removeHandler(handler)


def main():

	count = 5
	trainSize = [5]
	#splitSize = [5,10,15,20] #k
	splitSize = [5]
	#numSampleList = [50,100,150,200] #s
	numSampleList = [50]  # s
	maxDatasetSize = 20

	#Dataset File Names
	# datasetName = ['forestcover.arff', 'kdd.arff', 'pamap2.arff','powersupply.arff','sea.arff','syn002.arff', 'syn003.arff', 'mnist_100k_instances.data','news20_100k_instances.data']
	datasetName = ['forestcover.arff']
	#datasetName = ['PRSA_Data_Aotizhongxin_20130301-20170228.csv']

	#Directory of dataset
	basedir = 'D:\\STUDY\\Deeplearning\\paper&code\\deadline\\sampling_kmm-master(veryfastKMM)\\sampling_kmm-master\\'
	#basedir = '/root/Documents/scale-kmm/dataset/'

	mgr = Manager()

	for t in trainSize:
		mgr.start(count, t, splitSize, numSampleList, maxDatasetSize, datasetName, basedir)


if __name__ == '__main__':
	main()
