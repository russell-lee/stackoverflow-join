import sys
import re
import csv

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import numpy as np
import pickle
import time
import py_stringmatching as sm

numIterations = 50

HOLDOUT_BOOTSTRAP_RESULTS = 'results/bootstrap'
HOLDOUT_CLASSIFY_RESULTS = 'results/holdout'
OUTSAMPLE_CLASSIFY_RESULTS = 'results/outsample'

hasTwoNames = False
inSampleNames = 'NLMdata/dataCached/insample_abstracts_outfilesecondName'
outSampleNames = 'NLMdata/dataCached/outsample_abstracts_outfilesecondName'

def getWikiName(wID):
	afterBackslash = wID.split('/')[-1][:-1]
	return afterBackslash.replace('_',' ')

def getTitlesAndAbstracts(titlePairs,abstractPairs):
	tagNames = [t1 for (t1,t2) in titlePairs]
	wikiNames = [t2 for (t1,t2) in titlePairs]
	tagAbstracts = [a1 for (a1,a2) in abstractPairs]
	wikiAbstracts = [a2 for (a1,a2) in abstractPairs]
	return (tagNames,wikiNames,tagAbstracts,wikiAbstracts)

# Predicts unlabeled results and writes to csv.
def classifyAndPredictUnlabeled(fv_args,classifier,classifierName,folder,threshold = 0.5):
	insampleFV,insampleLabels,os_fv,os_abs,os_titles = fv_args
	full_model = classifier
	full_model.fit(insampleFV,insampleLabels)
	probs = full_model.predict_proba(os_fv)[:,1]
	predictions = [1 if prob > threshold else 0 for prob in probs]
	tagNames,wikiNames,tagExcerpts,wikiExcerpts = getTitlesAndAbstracts(os_titles,os_abs)	

	zipOutput = zip(tagNames,wikiNames,probs,predictions,tagExcerpts,wikiExcerpts)
	header = [["Tag Name", "Wiki Entity", "Probability","Predicted","Tag Excerpt","Wiki Excerpt"]]
	output = header + zipOutput
	fileTitle = folder + OUTSAMPLE_CLASSIFY_RESULTS + classifierName + '.csv'
	with open(fileTitle, 'wb') as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		for row in output:
			wr.writerow(row)
	print 'Unlabeled prediction results written to %s' %(fileTitle)

# Classifies on an 80/20 training split and returns the precision and recall on the holdout set.
def classify(fv_args,classifier,classifierName,folder,threshold = 0.5,saveResults = False):
	fv,labels,abstracts,titles = fv_args
	train,test,train_label,test_label,train_titles,test_titles,train_abstracts,test_abstracts = train_test_split(fv, labels,titles,abstracts, train_size = 0.8)
	model = classifier
	model.fit( train, train_label )
	probs = model.predict_proba( test )[:,1]
	predictions = [1 if prob > threshold else 0 for prob in probs]
	# tracking precision and recall
	precision,recall,fscore,_ = precision_recall_fscore_support(test_label, predictions, average = 'binary')
	if saveResults:
		# Write predictions on test set to CSV
		tagNames,wikiNames,tagExcerpts,wikiExcerpts = getTitlesAndAbstracts(test_titles,test_abstracts)
		header = [["Tag Name", "Wiki Entity", "Probability", "Predicted", "Actual", "Tag Excerpt", "Wiki Excerpt"]]
		zipOutput = zip(tagNames,wikiNames,probs,predictions,test_label,tagExcerpts,wikiExcerpts)
		output = header+zipOutput
		report = "Precision : %f , Recall : %f" % (precision,recall)
		fileTitle = folder + HOLDOUT_CLASSIFY_RESULTS + classifierName + '.csv'
		with open(fileTitle, 'wb') as myfile:
			wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
			wr.writerow([report])
			for row in output:
				wr.writerow(row)
		print 'Holdout results written to %s' % (fileTitle)
	return (precision,recall)

# Classifies over several iterations and returns the bootstrapped precision and recall.
def classifyNIterations(fv_args,classifier,classifierName,folder,threshold = 0.5):
	t0 = time.clock()
	iterValues = [classify(fv_args,classifier,classifierName,folder,threshold) for i in range(numIterations)]
	runtime = time.clock()-t0
	mean_values = np.mean(iterValues, axis = 0)
	std_values = np.std(iterValues, axis = 0)
	meanPrecision = mean_values[0]
	meanRecall = mean_values[1]
	stdPrecision = std_values[0]
	stdRecall = std_values[1]
	# Write bootstrapped results
	fileTitle = folder + HOLDOUT_BOOTSTRAP_RESULTS + classifierName + '.csv'
	with open(fileTitle,'wb') as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		output = [("Iterations",numIterations),
				  ("Classifier",classifierName),
				  ("Mean Precision",meanPrecision),
				  ("Std Precision",stdPrecision),
				  ("Mean Recall",meanRecall),
				  ("Std Recall",stdRecall),
				  ("Classify Runtime",runtime)]
		for row in output:
			wr.writerow(row)
		print "Bootstrapped holdout results written to %s" % (fileTitle)
	return(meanPrecision,meanRecall)

# Tracks precision and recall over a list of different decision thresholds.
def threshTest(fv_args,classifier,thresholdList):
	fv,labels,_,_ = fv_args
	train,test,train_label,test_label = train_test_split(fv, labels, train_size = 0.8)
	model = classifier
	model.fit( train, train_label )
	probs = model.predict_proba( test )[:,1]
	resultList = []
	for threshold in thresholdList:
		predictions = [1 if prob > threshold else 0 for prob in probs]
		precision,recall,_,_ = precision_recall_fscore_support(test_label, predictions, average = 'binary')
		resultList.append([threshold,precision,recall])
	return resultList	
if __name__ == '__main__':
	print "Shouldn't be doing anything."
