from scripts import FVComponent as FVC
from scripts import FVclassifier
import pickle
import time
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
import py_stringmatching as sm
import numpy as np
import os

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


INSAMPLE_FV_OUTFILE = 'dataCached/insampleFV_outfile'
OUTSAMPLE_FV_OUTFILE = 'dataCached/outsampleFV_outfile'
OUTSAMPLE_FV_REDUCED_OUTFILE = 'dataCached/outsampleFVreduced_outfile'

csAbstract = FVC.CosSim('CSAbs',TfidfVectorizer( ngram_range = ( 1, 3 ), sublinear_tf = True ),False)
csSentence = FVC.CosSim('CSSent',TfidfVectorizer( ngram_range = ( 1, 3 ), sublinear_tf = True ),True)
jacq3 = FVC.stringMatchExcerpts('FuzzJacc',sm.Jaccard(),sm.QgramTokenizer(qval=3,return_set = True))
cosM = FVC.stringMatchExcerpts('CosMeasure',sm.Cosine(),sm.WhitespaceTokenizer(return_set = True))
cosMq3 = FVC.stringMatchExcerpts('FuzzCosMeasure',sm.Cosine(),sm.QgramTokenizer(return_set = True))
LVdist = FVC.stringMatchTitles('LVDist',sm.Levenshtein())

DEFAULTFV = [jacq3,cosM,cosMq3,LVdist]
DEFAULTMODEL = LR()
DEFAULTMODELNAME = 'LogisiticRegression'
DEFAULTITERATIONS = 25


class join:
    def __init__(self,insampleData,outsampleData,dataFolder):
        self.insampleData = insampleData #pairs,labels,pairedAbstracts,pairedTitles
        self.outsampleData = outsampleData #pairs,labels,pairedAbstracts,pairedTitles
        self.dataFolder = dataFolder
        self.labels = insampleData[1]
        self.model = DEFAULTMODEL
        self.modelName = DEFAULTMODELNAME
        self.iterations = DEFAULTITERATIONS
        self.setComponentList(DEFAULTFV)
        self.hasTwoNames = False

    def __str__(self):
        return 'Feature Vector components: %s \nModel: %s' %(self.FVDescription,self.modelName)

    # Changes the component list and immediately updates all relevant variables.
    def setComponentList(self,cList):
        self.componentList = cList
        self.featureList = [feature for comp in self.componentList for feature in comp.features]
        self.FVDescription = '+'.join(fv.name for fv in self.componentList)
        self.insampleFVCache = self.dataFolder + self.FVDescription + '/' + INSAMPLE_FV_OUTFILE
        self.outsampleFVCache = self.dataFolder + self.FVDescription + '/' + OUTSAMPLE_FV_OUTFILE
        self.outsampleFVReducedCache = self.dataFolder + self.FVDescription + '/' + OUTSAMPLE_FV_OUTFILE
        ensure_dir(self.dataFolder + self.FVDescription + '/dataCached/')
        ensure_dir(self.dataFolder + self.FVDescription + '/results/')

    # Helper function for building feature vector.  Can optionally load from cache.
    def getFeatureVector(self,cacheDestination,cacheResults = True):
        self.featureList = [feature for comp in self.componentList for feature in comp.features]
        #self.featureList = [c.features for c in self.componentList]
        featureVector = np.stack(self.featureList, axis = -1)
        if cacheResults:
            pickle.dump(featureVector,open(cacheDestination,'wb'))
        return featureVector

    # Builds feature vector from component list.
    def buildFeatureVector(self,dataSource,cacheDestination,cacheResults = True):
        for c in self.componentList:
            c.generateFeatures(dataSource)
        return self.getFeatureVector(cacheDestination,cacheResults)

    # Builds full feature vector for insample (labeled) rows.
    def buildInsampleFV(self,cacheResults = True):
        self.insampleFV = self.buildFeatureVector(self.insampleData,self.insampleFVCache,cacheResults)
        self.classifyArgs = (self.insampleFV,self.labels,self.insampleData[2],self.insampleData[3])
        return self.insampleFV

    # Load insample feature vector from cache.
    def loadCachedInsampleFV(self):
        self.insampleFV = pickle.load(open(self.insampleFVCache,'rb'))
        self.classifyArgs = (self.insampleFV,self.labels,self.insampleData[2],self.insampleData[3])
        return self.insampleFV

    # Builds full feature vector for outsample (unlabeled) rows.  Typically takes a significant amount of time.  NLM unlabeled
    # dataset currently too large for any operation.
    def buildOutsampleFV(self,cacheResults = True):
        self.outsampleFV = self.buildFeatureVector(self.outsampleData,self.outsampleFVCache,cacheResults)
        self.predictArgs = (self.insampleFV,self.labels,self.outsampleFV,self.outsampleData[2],self.outsampleData[3])
        return self.outsampleFV

    # Builds proportions of outsample feature vector for some proportion between 0.0 to 1.0 that indicates how many of the rows should be kept.
    def buildOutsampleFVReduced(self,subSampleProportion,cacheResults = True):
        _,osPairs,_,osLabels,_,osAbs,_,osTitles = train_test_split(self.outsampleData[0],self.outsampleData[1],self.outsampleData[2],self.outsampleData[3], test_size = subSampleProportion)
        self.outsampleDataReduced = [osPairs,osLabels,osAbs,osTitles]
        self.outsampleFVReduced = self.buildFeatureVector(self.outsampleDataReduced,self.outsampleFVReducedCache,cacheResults)
        self.reducedPredictArgs = (self.insampleFV,self.labels,self.outsampleFVReduced,self.outsampleDataReduced[2],self.outsampleDataReduced[3])
        print len(self.insampleFV)
        print len(self.labels)
        return self.outsampleFVReduced

    def loadCachedOutsampleFV(self):
        self.outsampleFV = pickle.load(open(self.outsampleFVCache,'rb'))
        osAbs = self.outsampleData[2]
        osTitles = self.outsampleData[3]
        self.predictArgs = (self.insampleFV,self.labels,self.outsampleFV,osAbs,osTitles)
        return self.outsampleFV

    def loadCachedOutsampleFVReduced(self):
        self.outsampleFVReduced = pickle.load(open(self.outsampleFVReducedCache,'rb'))
        self.reducedPredictArgs = (self.insampleFV,self.labels,self.outsampleFVReduced,self.outsampleDataReduced[2],self.outsampleDataReduced[3])
        return self.outsampleFVReduced

    # Performs classification based on insample FV.  Make sure insample FV build or loaded from cache before executing.
    # Can perform classification on a proportion of insample FV from 0 to 1.0 (default is 1.0)
    def classify(self,threshold = 0.5,subSampleProportion = 1.0):
        t0=time.clock()
        if subSampleProportion == 1.0:
            precision,recall = FVclassifier.classify(self.classifyArgs,self.model,self.modelName,self.dataFolder + self.FVDescription + '/',threshold = threshold, saveResults = True)
            return (precision,recall,runtime)
        else:
            _,FV,_,labels,_,pairedAbs,_,pairedTitles = train_test_split(self.classifyArgs[0],self.classifyArgs[1],self.classifyArgs[2],self.classifyArgs[3], test_size = subSampleProportion)
            classifyArgs = (FV,labels,pairedAbs,pairedTitles)
            precision,recall = FVclassifier.classify(classifyArgs,self.model,self.modelName,self.dataFolder + self.FVDescription + '/',threshold = threshold, saveResults = True)
            runtime = time.clock()-t0
            return (precision,recall,runtime,len(labels))

    # Runs classification over several iterations and returns the bootstrapped precision and recall.
    # Can perform classification on a proportion of insample FV from 0 to 1.0 (default is 1.0)
    def classifyNIterations(self, threshold = 0.5,subSampleProportion = 1.0):
        t0=time.clock()

        if subSampleProportion == 1.0:
            precision,recall = FVclassifier.classifyNIterations(self.classifyArgs,self.model,self.modelName,self.dataFolder + self.FVDescription + '/',threshold = threshold)
        else:
            _,FV,_,labels,_,pairedAbs,_,pairedTitles = train_test_split(self.classifyArgs[0],self.classifyArgs[1],self.classifyArgs[2],self.classifyArgs[3], test_size = subSampleProportion)
            classifyArgs = (FV,labels,pairedAbs,pairedTitles)
            precision,recall = FVclassifier.classifyNIterations(classifyArgs,self.model,self.modelName,self.dataFolder + self.FVDescription + '/',threshold = threshold)
        #precision,recall = FVclassifier.classifyNIterations(self.classifyArgs,self.model,self.modelName,self.dataFolder + self.FVDescription + '/', threshold = threshold)
        runtime = time.clock()-t0
        return (precision,recall,runtime,len(labels))

    # Predicts on unlabeled pairs.  By default, predicts on the reduced proportion of unlabeled pairs.
    def predict(self,predictReduced = True, threshold = 0.5):
        if predictReduced:
            predictArgs = self.reducedPredictArgs
        else:
            predictArgs = self.predictArgs
        FVclassifier.classifyAndPredictUnlabeled(predictArgs,self.model,self.modelName,self.dataFolder + self.FVDescription + '/',threshold = threshold)

    # Test for determining precision and recall at different decision thresholds.
    def thresholdTest(self,thresholdList):
        return FVclassifier.threshTest(self.classifyArgs,self.model,thresholdList)

    def showFeatureVectorComponents(self):
        print self.FVDescription

    def quickExperiment(self):
        self.classify()
        return self.classifyNIterations()

    def fullExperiments(self):
        self.classify()
        self.classifyNIterations()
        self.predict(self.predictArgs)
