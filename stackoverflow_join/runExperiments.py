import join as myJoin
from scripts import FVComponent as FVC
import pickle

# Import preqrequisite packages to instantiate FVComponent instances
from sklearn.feature_extraction.text import TfidfVectorizer 
import py_stringmatching as sm

# Load preprocessed data from cache

SOInsampleFile = 'stackoverflowdata/dataCached/insample_abstracts_outfile'
SOOutsampleFile = 'stackoverflowdata/dataCached/outSample_abstracts_outfile'
SOInsampleData = pickle.load(open(SOInsampleFile,'rb'))
SOOutsampleData = pickle.load(open(SOOutsampleFile,'rb'))

nlmInsampleFile = 'NLMdata/dataCached/insample_abstracts_outfile'
nlmOutsampleFile = 'NLMdata/dataCached/outSample_abstracts_outfile'
nlmInsampleData = pickle.load(open(nlmInsampleFile,'rb'))
nlmOutsampleData = pickle.load(open(nlmOutsampleFile,'rb'))

# Instantiate FVComponent instances
csAbstract = FVC.CosSim('CSAbs',TfidfVectorizer( ngram_range = ( 1, 3 ), sublinear_tf = True ),False)
csSentence = FVC.CosSim('CSSent',TfidfVectorizer( ngram_range = ( 1, 3 ), sublinear_tf = True ),True)
cosM = FVC.stringMatchExcerpts('CosMeasure',sm.Cosine(),sm.WhitespaceTokenizer(return_set = True))
LVDist = FVC.stringMatchTitles('LVDist',sm.Levenshtein())

FVCList = [csAbstract,csSentence,cosM,LVDist]

def classifyAndPredict(insampleData,outsampleData,folderName,componentList):
    print len(insampleData[0])
    print len(outsampleData[1])
    # Declare instance of a join object with input arguments
    easyJoin = myJoin.join(insampleData,outsampleData,folderName)
    easyJoin.setComponentList(componentList)
    # Build feature vector
    easyJoin.buildInsampleFV()
    easyJoin.buildOutsampleFVReduced(0.01)
    # Classify and predict with logistic regression
    easyJoin.classify()
    easyJoin.classifyNIterations()
    easyJoin.predict()
# Load preprocessed data from cache

def main():
    classifyAndPredict(SOInsampleData,SOOutsampleData,'stackoverflowdata/',FVCList)
    classifyAndPredict(nlmInsampleData,nlmOutsampleData,'NLMdata/',FVCList)

if __name__ == '__main__':
    main()