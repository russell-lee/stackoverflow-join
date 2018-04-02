import sys

import nltk
from nltk.corpus import stopwords
import re
import random

import pickle

import csv

DISCARD_STOPWORDS = False
DISCARD_SPECIALCHARS = False
# Can set this variable to true if you would like to skip buildDomainData()
USE_CACHED_PREPROCESSING = False

WIKIDICT_OUTFILE = '../dataCached/wikidict_outfile'
# Files destinations for recording missing pairs
MISSING_INLINKS_FILE = '../errorLogs/missing_inlinks.csv'
MISSING_DICTIONARY_FILE = '../errorLogs/missing_dictionary.csv'




def preprocess(abstract):
    # Function to convert a raw abstract to a string of words
    # The input is a single string (an abstract), and 
    # the output is a single string (a preprocessed abstract)
    #
    if DISCARD_SPECIALCHARS:     
    	words = re.sub("[^a-zA-Z]", " ", abstract) 
    #
    # 2. Convert to lower case, split into individual words
    words = abstract.lower().split()                             

    if DISCARD_STOPWORDS:
    	stops = set(stopwords.words("english"))      
    	meaningful_words = [w for w in words if not w in stops]
    	return( " ".join( meaningful_words ))  
    else:
    	return( " ".join( words ))  
     
def buildWikiData():
	shortAbstractData = open(S_ABS_FILE,'r')
	shortAbstractArray = [[line.strip()[0:-8].split(" ",2)[0],line.strip()[1:-6].split(" ",2)[2][1:]] for line in shortAbstractData][1:] #eid, short abstract
	shortAbstractDictionary = {line[0].lower():preprocess(line[1]) for line in shortAbstractArray} # eid: short abstract
	with open(WIKIDICT_OUTFILE,'wb') as wikiFile:
		pickle.dump(shortAbstractDictionary,wikiFile)

def buildDomainData(tag_file,labeled_file,candidate_pairs_file,tagdict_outfile,processed_data_outfile):
	tagData = open(tag_file,'r')
	tagArray =[line.strip().split('\t') for line in tagData]

	tagDictionary = {line[0]:[preprocess(col) for col in line[1:]] for line in tagArray} # tagID: []

	labeledData = open(labeled_file,'r')
	labeledArray = [(line.strip().split('\t')[0],line.strip().split('\t')[1].lower()) for line in labeledData] #(id, wiki entity).
	labeledIDs = set([eID for (eID,wID) in labeledArray])

	# if there are additional fields that should be preprocessed in the candidate pairs, need a special case to handle keeping those fields.
	# NOTE: I did not finish implementing using both names for NLM data.
	candidatePairsData = open(candidate_pairs_file,'r')
	candidatePairsArray = [(line.strip().split('\t')[2],line.strip().split('\t')[1].lower()) for line in candidatePairsData]

	inSamplePairsArray = [(eID,wiki) for (eID,wiki) in candidatePairsArray if eID in labeledIDs]
	outOfSamplePairsArray = [(eID,wiki) for (eID,wiki) in candidatePairsArray if eID not in labeledIDs]

	with open(tagdict_outfile,'wb') as tagFile:
		pickle.dump(tagDictionary,tagFile)
	pickle_output = (labeledArray,inSamplePairsArray,outOfSamplePairsArray)
	with open(processed_data_outfile,'wb') as processFile:
		pickle.dump(pickle_output,processFile)

# Debugging functions for recording missing pairs and such
def logMissingInlinks(tagDict,fv_pairs,linked_pairs):
	missing_inlinks = [[t,tagDict[t][0],w] for (t,w) in linked_pairs if (t,w) not in set(fv_pairs)]
	print "%d out of the %d total inlink pairs are missing from the fv pairs" % (len(missing_inlinks), len(linked_pairs))
	print "Logging missing inlinks..."
	with open(MISSING_INLINKS_FILE, 'wb') as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		for row in missing_inlinks:
			wr.writerow(row)

def logMissingDictionaryData(tagIDs,wikiIDs,fv_pairs,linked_pairs):
	missing_linked_wiki = [w for (t,w) in linked_pairs if w not in wikiIDs]
	missing_fv_wiki = list(set([w for (t,w) in fv_pairs if w not in wikiIDs]))
	output = missing_linked_wiki+missing_fv_wiki

	print "Logging missing dictionary data..."

	with open(MISSING_DICTIONARY_FILE, 'wb') as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		for row in output:
			wr.writerow([row])

def getWikiName(wID):
	afterBackslash = wID.split('/')[-1][:-1]
	return afterBackslash.replace('_',' ')

def main():
	_,inputName,tagFilename,labeledFilename,candidatePairsFilename = sys.argv
	# Set file name variables
	# User-defined input files
	input_folder = '../' + inputName
	tag_file = input_folder + tagFilename
	labeled_file = input_folder + labeledFilename
	candidate_pairs_file = input_folder + candidatePairsFilename

	# Cached results output files
	processed_data_outfile = input_folder + 'dataCached/processed_outfile'
	tagdict_outfile = input_folder + 'dataCached/tagdict_outfile'

	abs_outfile = input_folder + 'dataCached/insample_abstracts_outfile'
	outsample_abs_outfile = input_folder + 'dataCached/outSample_abstracts_outfile'
	if USE_CACHED_PREPROCESSING:
		print "Using cached preprocessed data, skipping preprocessing..."
	else:
		print "Preprocessing data from scratch..."
		buildDomainData(tag_file,labeled_file,candidate_pairs_file,tagdict_outfile,processed_data_outfile)
		print "Preprocessing done!"

	print "Formatting Abstracts..."
	# load cached preprocessed data
	linked_pairs,inSample,outOfSample =pickle.load(open(processed_data_outfile,'rb'))


	tagDict = pickle.load(open(tagdict_outfile,'rb'))
	wikiDict = pickle.load(open(WIKIDICT_OUTFILE,'rb'))
	tagIDs = set(list(tagDict.keys()))
	wikiIDs = set(list(wikiDict.keys()))

	for i in inSample[0:10]:
		print i

	print 'heres the diff'

	for i in outOfSample[0:10]:
		print i
	# Only keep pairs that are actually dictionary keys
	linked_trim = [(t,w) for (t,w) in linked_pairs if (t in tagIDs and w in wikiIDs)]
	if len(linked_trim) == 0:
		quit()
	def buildAbstracts(fv_pairs,abs_outfileName,logMissingData = False, returnSelectPairs = False):
		# only keep pairs that can actually be looked up in the dictionary
		# TODO: verify that dictionary keys are correctly formatted.  most likely the wiki dictionary entries that are messed up
		fv_trim = [(t,w) for (t,w) in fv_pairs if (t in tagIDs and w in wikiIDs)]
		if returnSelectPairs:
			
			# Include linked pairs in feature vector
			output_pairs =list(set(fv_trim))
		else:
			# Include linked pairs in feature vector
			output_pairs = list(set(linked_trim+fv_trim))

		
		labels = [1 if (t,w) in linked_trim else 0 for (t,w) in output_pairs]
		pairedTitles = [(tagDict[tID][1],getWikiName(wID)) for (tID,wID) in output_pairs]
		pairedAbstracts = [(tagDict[tID][0],wikiDict[wID]) for (tID,wID) in output_pairs]

		pickle_output = (output_pairs,labels,pairedAbstracts,pairedTitles)
		# Cache results
		with open(abs_outfileName,'wb') as fp:
			pickle.dump(pickle_output,fp)

		if logMissingData:
			print "Cut %d linked pairs not in dictionary, %d found in dictionary" % (len(linked_pairs)-len(linked_trim),len(linked_trim))
			print "Cut %d fv pairs not in dictionary, %d found in dictionary" % (len(fv_pairs)-len(fv_trim),len(fv_trim))
			logMissingInlinks(tagDict,fv_pairs,linked_pairs)
			logMissingDictionaryData(tagIDs,wikiIDs,fv_pairs,linked_pairs)

		return pickle_output

	# only builds abstracts on labeled pairs
	labeled = buildAbstracts(inSample,abs_outfile)
	# builds on all unlabeled pairs
	unlabeled = buildAbstracts(outOfSample,outsample_abs_outfile, returnSelectPairs = True)
	print "Formatting done!"
	return (labeled,unlabeled)

if __name__ == '__main__':
	main()


