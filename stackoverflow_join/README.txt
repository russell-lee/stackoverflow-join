Stack Overflow - Wikipedia/NLM Join
Russell Lee, <mail.russelllee@gmail.com> May 2017

To preprocess data, do:
  $ python buildWikiData.py
  $ python buildAbstracts.py stackoverflowdata/ tagdata.tsv annotatedMatches.gp firstpass_joins.txt 
  $ python buildAbstracts.py NLMdata/ nlm-data.tsv labels.tsv candidatePairs.txt

Files and Folders:
  runExperiments.py is a simple script that does simple classification and predictions.  longExperiments.py is longer (and messier) and is capable of doing the experiments described in the writeup.
  \scripts contains the the scripts for preprocessing as well as the scripts called by join.py
  \NLMdata contains NLM input data as well as all the results of running the pipeline on NLM data.
  \stackoverflow is the same as above, but with stackoverflow(S.O.) data
  \data contains dbpedia data used for both S.O. and NLM pipelines.  Currently this is just short_abstracts_en.ttl
  \dataCached contains cached dbpedia data used for both S.O. and NLM pipelines.  Currently this is just wikidict_outfile

Output:

The output of classification/prediction will be a .csv file, but the folder it is written to varies depending on the pipeline workflow.
The folder structure will be something along the lines of \domainFolder\featuresList\results\outputType+classifier used

As an example, a workflow that used S.O. data, features of Cosine Similarity and Levenshtein Distance, Logistic Regression, and bootstrapped results would be written to the following file:
\stackoverflowdata\CSAbs+LVDist\results\bootstrapLogisticRegression.csv


Input formats: see writeup
