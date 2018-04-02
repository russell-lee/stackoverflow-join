import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity

class FVComponent():
	def __init__(self,name):
		self.name = name
		self.cacheFile = 'dataCached/FVcomponents/'
		self.features = None

	def generateFeatures(self,fv_input):
		pass

	def cacheFeatures(self,fv_input,folderName):
		self.features = self.generateFeatures(fv_input)
		destination = self.cacheFile + folderName + '/' + name
		pickle.dump(self.features,open(destination,'wb'))
		return self.features


	def loadFeatures(self,folderName):

		self.features = pickle.load(open(self.cacheFile,'rb'))
		return self.features



class CosSim(FVComponent):
	def __init__(self,name,tfidfVectorizer,firstSentenceOnly = False):
		FVComponent.__init__(self,name)
		self.vectorizer = tfidfVectorizer
		self.firstSentenceOnly = firstSentenceOnly

	def getCosineSimilarity(self,vec_Pair):
		sim_array = cosine_similarity(vec_Pair[0],vec_Pair[1])
		return sim_array[0][0]

	def generateFeatures(self,fv_input):
		_,_,pairedAbstracts,_ = fv_input
		consecutiveAbstracts = []
		for i in range(len(pairedAbstracts)):
			(tagAbs,wikiAbs) = pairedAbstracts[i]

			consecutiveAbstracts.append(tagAbs)
			consecutiveAbstracts.append(wikiAbs)

		if self.firstSentenceOnly:
			def getFirstSentence(abstract):
				#TODO check why i am using a comma and not a period?
				return abstract.partition(',')[0]
			pairedData = [getFirstSentence(abstract) for abstract in consecutiveAbstracts]
		else:
			pairedData = consecutiveAbstracts

		tfidf_data = self.vectorizer.fit_transform(pairedData)
		tfidf_data_paired = [(tfidf_data[2*i],tfidf_data[2*i+1]) for i in range(len(pairedAbstracts))]
		abstracts_cosSim = [self.getCosineSimilarity(pair) for pair in tfidf_data_paired]
		self.features = np.array(abstracts_cosSim)
		return self.features

class stringMatchTitles(FVComponent):
	def __init__(self,name,simFunc):
		FVComponent.__init__(self,name)
		self.simFunc = simFunc

	def generateFeatures(self,fv_input):
		_,_,_,pairedTitles = fv_input
		try:
			self.features = np.array([self.simFunc.get_sim_score(t1,t2) for (t1,t2) in pairedTitles])
			return self.features
		except AttributeError:
			# Similarity function has no normalized score, giving raw scores instead
			self.features = np.array([self.simFunc.get_raw_score(t1,t2) for (t1,t2) in pairedTitles])
			return self.features
			


class stringMatchExcerpts(FVComponent):
	def __init__(self,name,simFunc,tokenizer):
		FVComponent.__init__(self,name)
		self.simFunc = simFunc
		self.tokenizer = tokenizer

	def generateFeatures(self,fv_input):
		_,_,pairedAbstracts,_ = fv_input
		tokenizedExcerpts = [(self.tokenizer.tokenize(e1),self.tokenizer.tokenize(e2)) for (e1,e2) in pairedAbstracts]
		try:
			self.features = np.array([self.simFunc.get_sim_score(e1,e2) for (e1,e2) in tokenizedExcerpts])
			return self.features
		except AttributeError:
			# Similarity function has no normalized score, giving raw scores instead
			self.features = np.array([self.simFunc.get_raw_score(e1,e2) for (e1,e2) in tokenizedExcerpts])
			return self.features

def main():
	import pickle
	import py_stringmatching as sm
	from sklearn.feature_extraction.text import TfidfVectorizer 
	INSAMPLE_ABS_OUTFILE = '../dataCached/insample_abstracts_outfile'
	OUTSAMPLE_ABS_OUTFILE = '../dataCached/outSample_abstracts_outfile'
	OUTSAMPLE_ABS_REDUCED_OUTFILE = '../dataCached/outSample_abstracts_reduced_outfile'
	a1 = pickle.load(open(INSAMPLE_ABS_OUTFILE,'rb'))
	a2 = pickle.load(open(OUTSAMPLE_ABS_OUTFILE,'rb'))
	a3 = pickle.load(open(OUTSAMPLE_ABS_REDUCED_OUTFILE,'rb'))
	csAbstract = CosSim('Cos Sim Abstract',TfidfVectorizer( ngram_range = ( 1, 3 ), sublinear_tf = True ),False)
	csSentence = CosSim('Cos Sim Sentence',TfidfVectorizer( ngram_range = ( 1, 3 ), sublinear_tf = True ),True)
	jacq3 = stringMatchExcerpts('Fuzzy Jaccard',sm.Jaccard(),sm.QgramTokenizer(qval=3))
	
	components = [csAbstract,csSentence,jacq3]
	a1Features = [c.generateFeatures(a1) for c in components]
	print len(a1Features)

if __name__ == '__main__':
	print 'Testing functionality...'
	main()