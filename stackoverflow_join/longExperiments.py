import join as myJoin
from scripts import FVComponent as FVC
import pickle
import csv

#import models
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression as LR

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer 
import py_stringmatching as sm

insample_data = 'dataCached/insample_abstracts_outfile'
outsample_data = 'dataCached/outSample_abstracts_outfile'

nlmInsampleFile = 'NLMdata/' + insample_data
nlmOutsampleFile = 'NLMdata/' + outsample_data
nlmInsampleData = pickle.load(open(nlmInsampleFile,'rb'))
nlmOutsampleData = pickle.load(open(nlmOutsampleFile,'rb'))



nlmTwoNamesInsample = pickle.load(open(nlmInsampleFile + 'secondName','rb'))
nlmTwoNamesOutsample = pickle.load(open(nlmOutsampleFile + 'secondName','rb'))

SOInsampleFile = 'stackoverflowdata/' + insample_data
SOOutsampleFile = 'stackoverflowdata/' + outsample_data
SOInsampleData = pickle.load(open(SOInsampleFile,'rb'))
SOOutsampleData = pickle.load(open(SOOutsampleFile,'rb'))

csAbstract = FVC.CosSim('CSAbs',TfidfVectorizer( ngram_range = ( 1, 3 ), sublinear_tf = True ),False)
csSentence = FVC.CosSim('CSSent',TfidfVectorizer( ngram_range = ( 1, 3 ), sublinear_tf = True ),True)
jac = FVC.stringMatchExcerpts('Jacc',sm.Jaccard(),sm.WhitespaceTokenizer(return_set = True))
jacq3 = FVC.stringMatchExcerpts('FuzzJacc',sm.Jaccard(),sm.QgramTokenizer(qval=3,return_set = True))
dice = FVC.stringMatchExcerpts('Dice',sm.Dice(),sm.WhitespaceTokenizer(return_set = True))
diceq3 = FVC.stringMatchExcerpts('Dice',sm.Dice(),sm.QgramTokenizer(qval = 3, return_set = True))
cosM = FVC.stringMatchExcerpts('CosMeasure',sm.Cosine(),sm.WhitespaceTokenizer(return_set = True))
cosMq3 = FVC.stringMatchExcerpts('FuzzCosMeasure',sm.Cosine(),sm.QgramTokenizer(return_set = True))
LVdist = FVC.stringMatchTitles('LVDist',sm.Levenshtein())
sw = FVC.stringMatchTitles('SW',sm.SmithWaterman())
nw = FVC.stringMatchTitles('NW',sm.NeedlemanWunsch())
jw = FVC.stringMatchTitles('JW',sm.JaroWinkler())


def writeToCSV(fileName,header,tableList):
    wr = csv.writer(open(fileName,'wb'), quoting=csv.QUOTE_ALL)
    wr.writerow(header)
    for row in tableList:
        wr.writerow(row)


# Given a set of feature vector components, records precision and recall over several 
# classifiers.  Records output to a table and vertical bar plot.

def modelExperiment(insampleData,outsampleData,dataFolder,componentList,models,modelNames,tableFile,plotFile,buildFV = True):
    j1 = myJoin.join(insampleData,outsampleData,dataFolder)
    j1.setComponentList(componentList)
    if buildFV:
        j1.buildInsampleFV()
    else:
        j1.loadCachedInsampleFV()
    modelResults = []
    for (mod,modName) in zip(models,modelNames):
        j1.model = mod
        j1.modelName = modName
        precision,recall,runtime = j1.quickExperiment()
        modelResults.append([modName,precision,recall,runtime])

    # Write summary of results to csv table
    writeToCSV(dataFolder + tableFile, ['','Precision','Recall','Runtime'], modelResults)


    # Write summary of results to plot
    precisionList = [res[1] for res in modelResults]
    recallList = [res[2] for res in modelResults]
    runtimeList = [res[3] for res in modelResults]
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(len(models))
    bar_width = 0.35
    opacity = 0.8
     
    rects1 = plt.bar(index, tuple(precisionList), bar_width,
                     alpha=opacity,
                     color='b',
                     label='Precision')
     
    rects2 = plt.bar(index + bar_width, tuple(recallList), bar_width,
                     alpha=opacity,
                     color='g',
                     label='Recall')
     
    plt.xlabel('Classifier')
    plt.ylabel('Scores')
    plt.title('Bootstrapped Precision and Recall Scores vs. Classifier')
    plt.xticks(index + bar_width, tuple(modelNames))
    plt.legend()
    plt.tight_layout()
    plt.savefig(dataFolder + plotFile)

def thresholdExperiment(insampleData,outsampleData,dataFolder,allComponents,model,modelName,tableFile):
    thresholdRange = np.arange(0.0,0.01,1.01)
    j1 = myJoin.join(insampleData,outsampleData,dataFolder)
    j1.model = model
    j1.modelName = modelName
    j1.setComponentList(allComponents)
    j1.loadCachedInsampleFV()
    expResults = []
    for tHold in thresholdRange:
        precision,recall,runtime = j1.classifyNIterations(tHold)
        expResults.append([tHold,precision,recall,runtime])
        print tHold

    writeToCSV(dataFolder + tableFile,['Threshold','Precision','Recall','Runtime'], expResults )


def featureVectorExperiment(insampleData,outsampleData,dataFolder,allComponents,model,modelName,tableFile,plotFile):
    j1 = myJoin.join(insampleData,outsampleData,dataFolder)
    j1.model = model
    j1.modelName = modelName
    FVResults = []
    FVNames = []
    for componentList in allComponents:
        j1.setComponentList(componentList)
        j1.buildInsampleFV()
        precision,recall,runtime = j1.quickExperiment()
        FVNames.append(j1.FVDescription)
        FVResults.append([j1.FVDescription,precision,recall,runtime])

    # TODO this can be written to a function
    # Write summary of results to csv table
    wr = csv.writer(open(dataFolder + tableFile,'wb'), quoting=csv.QUOTE_ALL)
    header = ['','Precision','Recall','Runtime']
    wr.writerow(header)
    for row in FVResults:
        wr.writerow(row)

    # Write summary of results to plot
    precisionList = [res[1] for res in FVResults]
    recallList = [res[2] for res in FVResults]
    runtimeList = [res[3] for res in FVResults]
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(len(allComponents))
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, tuple(precisionList), bar_width,
                     alpha=opacity,
                     color='b',
                     label='Precision')
     
    rects2 = plt.bar(index + bar_width, tuple(recallList), bar_width,
                     alpha=opacity,
                     color='g',
                     label='Recall')
     
    plt.xlabel('Feature Vector')
    plt.ylabel('Scores')
    plt.title('Bootstrapped Precision and Recall Scores using %s vs. Feature Vector' %(modelName))
    plt.xticks(index + bar_width, tuple(FVNames))
    plt.legend()
    plt.tight_layout()
    plt.savefig(dataFolder + plotFile)


def main():
    fullFV = [csAbstract,csSentence,jac,jacq3,dice,diceq3,cosM,cosMq3,LVdist,sw,nw,jw]
    fullModels = [LR(),DT(),KNC(),RF(n_estimators = 200),ABC(),GNB(),QDA()]
    fullModelNames = ['LogisticRegression','DTree','KNN','RandomForest','AdaBoosted','GaussianNB','QuadraticDiscriminantAnalysis']
    #modelExperiment(nlmInsampleData,nlmOutsampleData,'NLMdata/',fullFV,[LR(),DT(),KNC(),RF(),ABC(),GNB(),QDA()],
    #                ['LogisticRegression','DTree','KNN','RandomForest','AdaBoosted','GaussianNB','QuadraticDiscriminantAnalysis'],
    #                'NLMmodelExperiment1.csv','NLMclassifier_plot1.png',True)

    def SOmodelexp1():
        modelExperiment(SOInsampleData,SOOutsampleData,'stackoverflowdata/',fullFV,[LR(),DT(),KNC(),RF(n_estimators = 200),ABC(),GNB(),QDA()],
                        ['LogisticRegression','DTree','KNN','RandomForest','AdaBoosted','GaussianNB','QuadraticDiscriminantAnalysis'],
                        'SOmodelExperiment1.csv','SOclassifier_plot1.png',True)
    def SOmodelexp2():
        modelExperiment(SOInsampleData,SOOutsampleData,'stackoverflowdata/',fullFV,[LR(),RF(n_estimators = 200),ABC()],
                        ['LogisticRegression','RandomForest','AdaBoosted'],
                        'SOmodelExperiment2.csv','SOclassifier_plot2.png',True)
    def NLMmodelexp1():
        modelExperiment(nlmInsampleData,nlmOutsampleData,'NLMdata/',fullFV,[LR(),DT(),KNC(),RF(),ABC(),GNB(),QDA()],
                    ['LogisticRegression','DTree','KNN','RandomForest','AdaBoosted','GaussianNB','QuadraticDiscriminantAnalysis'],
                    'NLMmodelExperiment1.csv','NLMclassifier_plot1.png',True)

    #featureVectorExperiment(SOInsampleData,SOOutsampleData,'stackoverflowdata/',[[jacq3],[cosM],[cosMq3],[jacq3,cosM],[cosM,cosMq3],[csAbstract]],DT(),
    #                'DTree','SOFVExperiment1.csv','SOFV_plot1.png')


    def NLMexperiments():
        j2 = myJoin.join(nlmInsampleData,nlmOutsampleData,'NLMdata/')
        j2.setComponentList(fullFV)
        j2.loadCachedInsampleFV()
        results = []
        for prop in np.arange(0.05,0.25,0.01):
            precision,recall,_,size = j2.classifyNIterations(subSampleProportion = prop)
            results.append([size,precision,recall])
        writeToCSV('NLMdata/sizeTest2.csv',['Size','Precision','Recall'], results)


    def SOexperiments():
        j1 = myJoin.join(SOInsampleData,SOOutsampleData,'stackoverflowdata/')
        j1.setComponentList(fullFV)
        j1.buildInsampleFV()
        j1.model = RF(n_estimators = 200)
        j1.modelName = 'RF'
        def threshHoldTest():
            singleThreshTest = j1.thresholdTest(np.arange(0.0,1.01,0.01))
            writeToCSV('stackoverflowdata/simpleThresholdTest1.csv',['Threshold','Precision','Recall'], singleThreshTest )
            print 'simple thing done'
            fiftyThreshTest = [j1.thresholdTest(np.arange(0.0,1.01,0.01)) for i in range(50)]
            mean_values = np.mean(fiftyThreshTest, axis = 0)
            writeToCSV('stackoverflowdata/fiftyThresholdTest1.csv',['Threshold','Precision','Recall'], mean_values )
            print 'fifty thing done'
            thresholdExperiment(SOInsampleData,SOOutsampleData,'stackoverflowdata/',fullFV,RF(n_estimators = 200),'RF','thresholdExperiment1.csv')

    #SOexperiments()
    NLMexperiments()
    



    
if __name__ == '__main__':
    main()