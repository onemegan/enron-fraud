# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 19:31:19 2017

@author: ICAR20
"""

#!/usr/bin/python
import os
os.getcwd()
os.chdir('C:/Dropbox/Pers/Courses/Nanodegree/P5_MachineLearning/ud120-projects/final_project')

import sys
sys.path.append("../tools/")
import pickle
import matplotlib.pyplot
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from pprint import pprint


import os
os.getcwd()
os.chdir('C:/Dropbox/Pers/Courses/Nanodegree/P5_MachineLearning/ud120-projects/final_project')


### STEP 1: SELECT FEATURES TO USE. ###

#Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

features_list = data_dict[data_dict.keys()[0]].keys()

### STEP 2: REMOVE OUTLIERS ###
# Calculate the percnetage of NaNs in dataset for each feature
def percent_nans(features_list):
    feature_nans = {}      
    for feature in features_list:
        feature_nan_count = 0          
        for key in data_dict.keys():
            if data_dict[key][feature]=='NaN':
                feature_nan_count += 1
        feature_nans[feature] = round(float(feature_nan_count)/len(data_dict)
        ,2)
    return feature_nans
print("NaN Percentages:")
pprint(percent_nans(features_list))

# Identify outliers of salary and bonus using visualization
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# Now remove the identified outlier which is the 'TOTAL' data point
data_dict.pop('TOTAL', 0)

# If an individual has zero values for all features, its an outlier.
# Identify the data points:
zero_keys = []
for key in data_dict:
    n = 0
    for ftr in features_list:
        if data_dict[key][ftr] == 'NaN':
            n += 1
    if n == len(features_list) - 1: # excluding the 'poi' key
        zero_keys.append(key)
print("\nData Points that Have NaN's for All Features:")
print zero_keys, '\n'  # 'LOCKHART EUGENE E'

print data_dict['LOCKHART EUGENE E']
# Remove this individual
for key in zero_keys:
    data_dict.pop(key, 0)
  
### STEP 3: CREATE NEW FEATURES ###
 
# Store to my_dataset for easy export below.
my_dataset = data_dict
len(my_dataset.keys())

# Create two new features, "fraction_to_poi" and "fraction_from_poi" which
# represent the proportion of emails each individual sends or receives to POIs.
my_dataset = data_dict

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    fraction = 0.
    if poi_messages != "NaN":
        if all_messages != "NaN":
            fraction = float(poi_messages)/all_messages
    return fraction

def computeAvgFraction(fraction_to_poi, fraction_from_poi):
    fraction = 0
    if fraction_to_poi != 0.:
        if fraction_from_poi != 0.:
            fraction = (fraction_to_poi + fraction_from_poi)/2
    return fraction

# Add new features to my_dataset
submit_dict = {}
for name in my_dataset:
    data_point = my_dataset[name]
    print name
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    print fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    print fraction_to_poi
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi

    ## Creating overall fraction based on weighted average    
    avg_fraction_to_from_poi = computeAvgFraction(fraction_to_poi, 
                                                  fraction_from_poi)
    #(fraction_to_poi*float(to_messages)+fraction_from_poi*float(from_messages))/float(to_messages+from_messages)
    print avg_fraction_to_from_poi
    data_point["avg_fraction_to_from_poi"] = avg_fraction_to_from_poi   
    
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi,
                       "avg_fraction_to_from_poi":avg_fraction_to_from_poi}
#my_dataset[my_dataset.keys()[0]]


# Update features list - include fraction but remove total to from
all_features = my_dataset[my_dataset.keys()[0]].keys()

features_list = all_features
    # remove email address
features_list.remove("email_address")
len(features_list)
order=[3,0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
features_list = [ features_list[i] for i in order]
print features_list

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### STEP 4: MAKE THE ESTIMATOR ###

# Prepare the pipeline including preprocessing, feature selection and algorithm running
pipe = make_pipeline(
    MinMaxScaler(),
    SelectKBest(),
    SVC()
    )
 
params = {
    'selectkbest__k': [1,2,3,4,5,6, 7, 8, 9, 10, 11, 12, 13, 14],
    'selectkbest__score_func': [f_classif],
    'svc__C': [0.1, 1, 10, 100],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': [0.001, 0.0001],
    }

# Split the data into training and testing sets
features_train, features_test, labels_train, labels_test =  train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42
    )

# Make an StratifiedShuffleSplit iterator for cross-validation in GridSearchCV
sss = StratifiedShuffleSplit(
    labels_train,
    n_iter = 20,
    test_size = 0.5,
    random_state = 0
    )

# Make the estimator using GridSearchCV and run cross-validation
# Algorithms: SVC(), LinearSVC(), KNeighborsClassifier(), RandomForestClassifier(), GaussianNB()
print 'GridSearching with cross-validation...'
clf = GridSearchCV(
    pipe,
    param_grid = params,#dictionary of parameters and possible settings to try
    scoring = 'f1',
    n_jobs = 1, # number of jobs to run in parallel
    cv = sss, #determines cross validation splitting strategy
    verbose = 1,
    error_score = 0
    )


### STEP 5: MODEL FITTING AND TESTING ###

# Fit the model using premade estimator clf
clf.fit(features_train, labels_train)

# Get the selected features
#https://discussions.udacity.com/t/how-to-find-out-the-features-selected-by-selectkbest/45118/8
pipe.fit(features_train, labels_train)
features_k= clf.best_params_['selectkbest__k']
SKB_k=SelectKBest(f_classif, k=features_k)
SKB_k.fit_transform(features_train,labels_train) # data used in gridsearch  
features_selected=[features_list[1:][i]for i in SKB_k.get_support(indices=True)]
print "Selected Features", features_selected

feature_scores = SKB_k.scores_
features_scores_selected=[feature_scores[i]for i in SKB_k.get_support(indices=True)]
print 'Feature Scores', features_scores_selected

# Test the model using the hold-out test data
pred = clf.predict(features_test)
print '\n', "Classification Performance Report:"
print(classification_report(labels_test, pred))
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html


### STEP 6: GENERATE THE PICKLE FILES ###

dump_classifier_and_data(clf, my_dataset, features_list)