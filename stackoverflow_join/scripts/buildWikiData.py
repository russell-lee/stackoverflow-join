import re
import nltk
import pickle
from nltk.corpus import stopwords


DISCARD_STOPWORDS = False
DISCARD_SPECIALCHARS = False


# Fixed input files
S_ABS_FILE = '../data/short_abstracts_en.ttl'
WIKIDICT_OUTFILE = '../dataCached/wikidict_outfile'



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

def main():
	shortAbstractData = open(S_ABS_FILE,'r')
	shortAbstractArray = [[line.strip()[0:-8].split(" ",2)[0],line.strip()[1:-6].split(" ",2)[2][1:]] for line in shortAbstractData][1:] #eid, short abstract
	shortAbstractDictionary = {line[0].lower():preprocess(line[1]) for line in shortAbstractArray} # eid: short abstract
	with open(WIKIDICT_OUTFILE,'wb') as wikiFile:
		pickle.dump(shortAbstractDictionary,wikiFile)

if __name__ == '__main__':
	main()


