import sys
import re
import csv


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity


import numpy as np
import pickle
 
# Input files
WIKIDICT_OUTFILE = 'dataCached/wikidict_outfile'
TAGDICT_OUTFILE = 'dataCached/tagdict_outfile'
CATDICT_OUTFILE = 'dataCached/catdict_outfile'

INSAMPLE_ABS_OUTFILE = 'dataCached/insample_abstracts_outfile'
OUTSAMPLE_ABS_OUTFILE = 'dataCached/outSample_abstracts_outfile'
OUTSAMPLE_ABS_REDUCED_OUTFILE = 'dataCached/outSample_abstracts_reduced_outfile'

INSAMPLE_STRINGMATCH_OUTFILE = 'dataCached/stringmatch_outfile'
OUTSAMPLE_STRINGMATCH_OUTFILE = 'dataCached/outSample_stringmatch_outfile'
OUTSAMPLE_STRINGMATCH_REDUCED_OUTFILE = 'dataCached/outSample_stringmatch_reduced_outfile'

# Output files
INSAMPLE_FV_OUTFILE = 'dataCached/insampleFV_outfile'
OUTSAMPLE_FV_OUTFILE = 'dataCached/outsampleFV_outfile'
OUTSAMPLE_FV_REDUCED_OUTFILE = 'dataCached/outsampleFVreduced_outfile'
FEATURE_NAMES_OUTFILE = 'dataCached/fv_names_outfile'

# Debugging results files
COSSIM_TEST = 'cossimtest.csv'
SENTENCE_TEST = 'sentence.csv'

# Open cached files
with open(CATDICT_OUTFILE,'rb') as catDictFile, open(WIKIDICT_OUTFILE,'rb') as wikiFile, open(TAGDICT_OUTFILE,'rb') as tagFile:
	CATDICT= pickle.load(catDictFile)
	TAGDICT = pickle.load(tagFile)
	WIKIDICT = pickle.load(wikiFile)

with open(INSAMPLE_ABS_OUTFILE,'rb') as insampleFile, open(OUTSAMPLE_ABS_OUTFILE,'rb') as outsampleFile, open(OUTSAMPLE_ABS_REDUCED_OUTFILE,'rb') as outsampleReducedFile:
	insampleAbs = pickle.load(insampleFile)
	outsampleAbs = pickle.load(outsampleFile)
	outsampleAbsReduced = pickle.load(outsampleReducedFile)


buildInsample = True
buildOutsample = False
buildOutsampleReduced = True


def dictLookup(wID):
	if wID in CATDICT:
		return CATDICT[wID]
	else:
		return ''

# generates portions of feature vector according to input arguments
def buildFeatureVector(fv_input,fv_args,stringMatchTuple):
	fv_pairs,labels,abstracts = fv_input
	BOWVectorizer = fv_args[0]
	categoryVocabVectorizer = fv_args[1]
	cossimVectorizer = fv_args[2]
	cossimSentenceVectorizer = fv_args[3]
	doStringmatch = fv_args[4]
	fv_components = []
	fv_names = []

	if doStringmatch:
		# document level string matched arrays
		abs_jac,abs_jacq3,abs_dice,abs_diceq3,abs_cos,abs_cosq3,titleLVdist,swScore,nwScore = stringMatchTuple
		fv_components.append(abs_jac)
		fv_components.append(abs_jacq3)
		fv_components.append(abs_dice)
		fv_components.append(abs_diceq3)
		fv_components.append(abs_cos)
		fv_components.append(abs_cosq3)

		fv_names.append('Jaccard')
		fv_names.append('Jaccard Q3')
		fv_names.append('Dice')
		fv_names.append('Dice Q3')
		fv_names.append('Cosine Measure')
		fv_names.append('Cosine Measure Q3')


		#title level string matching
		fv_components.append(titleLVdist)
		fv_components.append(swScore)
		fv_components.append(nwScore)
		fv_names.append('Title Edit Distance')
		fv_names.append('Title Smith Waterman Score')
		fv_names.append('Title Needleman Wunsch Score')

	if BOWVectorizer:
		BOW_data = BOWVectorizer.fit_transform(abstracts)
		fv_components.append(BOW_data)
	# if either part of cosine similarity is required, set up necessary helper functions
	# and data structures
	if cossimVectorizer or cossimSentenceVectorizer:
		def getCosineSimilarity(vec_Pair):
			sim_array = cosine_similarity(vec_Pair[0],vec_Pair[1])
			return sim_array[0][0]

		abstracts_consecutive_pair = []
		for i in range(len(fv_pairs)):
			(tID,wID) = fv_pairs[i]
			abstracts_consecutive_pair.append(TAGDICT[tID][2])
			abstracts_consecutive_pair.append(WIKIDICT[wID])

		# debugging to make sure the consecutive pairing makes sense	
		abstracts_tupled = [(abstracts_consecutive_pair[2*i],abstracts_consecutive_pair[2*i+1]) for i in range(len(fv_pairs))]

		if cossimVectorizer:
			tfidf_data = cossimVectorizer.fit_transform(abstracts_consecutive_pair)
			tfidf_data_paired = [(tfidf_data[2*i],tfidf_data[2*i+1]) for i in range(len(fv_pairs))]
			abstracts_cosSim = [getCosineSimilarity(pair) for pair in tfidf_data_paired]
			fv_components.append(np.array(abstracts_cosSim))
			fv_names.append("Cosine Similarity")

			


			# write results to see if cosine similarity is working
			def writeCosineSim():
				cossim_output = zip(fv_pairs,abstracts_cosSim,labels)
				with open(COSSIM_TEST, 'wb') as myfile:
					wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
					for row in cossim_output:
						wr.writerow(row)
				print 'done with cossim test'

			# record cosine similarity for a given pair when building prediction rows

			# writeCosineSim()
			# TESTING: cossim with category information as well
			# cosine similarity between categories and tag abstract
			if categoryVocabVectorizer:
				cat_consecutive_pair = []
				for i in range(len(fv_pairs)):
					(tID,wID) = fv_pairs[i]
					cat_consecutive_pair.append(TAGDICT[tID][2])
					cat_consecutive_pair.append(dictLookup(wID).replace('\n', ' '))
				cat_tfidf_data = cossimVectorizer.fit_transform(cat_consecutive_pair)
				cat_tfidf_data_paired = [(cat_tfidf_data[2*i],cat_tfidf_data[2*i+1]) for i in range(len(fv_pairs))]
				cat_cosSim = [getCosineSimilarity(pair) for pair in cat_tfidf_data_paired]
				fv_components.append(np.array(cat_cosSim))
				fv_names.append("Cosine Similarity with Categories")


		if cossimSentenceVectorizer:
			def getFirstSentence(abstract):
				# for some reason a comma was being used??
				return abstract.partition(',')[0]

			sentence_paired = [getFirstSentence(abstract) for abstract in abstracts_consecutive_pair]
			tfidf_sentence_data = cossimSentenceVectorizer.fit_transform(sentence_paired)
			tfidf_sentence_data_paired = [(tfidf_sentence_data[2*i],tfidf_sentence_data[2*i+1]) for i in range(len(fv_pairs))]
			sentence_cosSim = [getCosineSimilarity(pair) for pair in tfidf_sentence_data_paired]
			fv_components.append(np.array(sentence_cosSim))
			fv_names.append("Cosine Similarity in 1st Sentence")
	
	full_FV = np.stack(fv_components,axis = -1)
	# return values are kind of cumbersome
	if buildInsample:
		pickle.dump(fv_names,open(FEATURE_NAMES_OUTFILE,'wb'))
	return (full_FV,labels,fv_pairs,TAGDICT)

def main():
	print "Shouldn't be running anything"

if __name__ == '__main__':
	main()

def build(fv_args):
	fvTitle = ['Insample', 'Outsample', 'Outsample Reduced']
	absToBuild = [buildInsample,buildOutsample,buildOutsampleReduced]
	absList = [insampleAbs,outsampleAbs,outsampleAbsReduced]
	cacheList = [INSAMPLE_FV_OUTFILE,OUTSAMPLE_FV_OUTFILE,OUTSAMPLE_FV_REDUCED_OUTFILE]


	with open(INSAMPLE_STRINGMATCH_OUTFILE,'rb') as sm1, open(OUTSAMPLE_STRINGMATCH_OUTFILE,'rb') as sm2, open(OUTSAMPLE_STRINGMATCH_REDUCED_OUTFILE,'rb') as sm3:
		insampleStringmatch = pickle.load(sm1)
		outsampleStringmatch = pickle.load(sm2)
		outsampleStringmatchReduced = pickle.load(sm3)
	smList = [insampleStringmatch,outsampleStringmatch,outsampleStringmatchReduced]

	returnOutput = []
	for (title,toBuild,absInput,cacheFile,stringMatch) in zip(fvTitle,absToBuild,absList,cacheList,smList):
		if toBuild:
			print "Building %s FV" % (title)
			cacheOutput = buildFeatureVector(absInput,fv_args,stringMatch)
			pickle.dump(cacheOutput,open(cacheFile,'wb'))
			returnOutput.append(cacheOutput)
		else:
			print "%s FV skipped" % (title)
			returnOutput.append([])

	return tuple(returnOutput)







