# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 12:08:23 2018

@author: atl30511
"""

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt


#Create your df here:
os.chdir('C:/Users/jason/Downloads/ml capstone/')
raw_data = pd.read_csv("profiles.csv")

#look at all the columns
raw_data.columns

#remove essay fields
raw_data.drop(['essay0','essay1', 'essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9'], axis=1, inplace=True)




"""
These lifestyle fields seem interesting. I think may be some predictive power between them
for example someone who has a strict diet might not drink/smoke/do drugs. They are typically very strict about what they put in their bodies
or someone in the tech industry would be more likely to do drugs than someone in finance/banking

#lets see if we can predict drug use based on some of these other lifestyle/social fields

"""

#graph some of the columns
raw_data['drugs_clean']=raw_data.drugs.replace(np.nan, 'No Response', regex=True)
values = raw_data.drugs_clean.value_counts()
x = list(values.index)
y = list(values)
plt.bar(x,y)
plt.xlabel('Drug Usage')
plt.ylabel('Frequency')
plt.title('Count of Drugs Survey')
plt.show()


raw_data['drinking_clean']=raw_data.drinks.replace(np.nan, 'No Response', regex=True)
values = raw_data.drinking_clean.value_counts()
x = list(values.index)
y = list(values)
plt.bar(x,y)
plt.xlabel('Drinks Responses')
plt.xticks(rotation=45)
plt.ylabel('Frequency')
plt.title('Count of Drinks Survey')
plt.show()

raw_data['smoking_clean']=raw_data.smokes.replace(np.nan, 'No Response', regex=True)
values = raw_data.smoking_clean.value_counts()
x = list(values.index)
y = list(values)
plt.bar(x,y)
plt.xlabel('Smokes Responses')
plt.xticks(rotation=45)
plt.ylabel('Frequency')
plt.title('Count of Smokes Survey')
plt.show()


#replace Nan with a text value to keep track of how many chose not to answer
#raw_data['drugs']=raw_data.drugs.replace(np.nan, 'no response', regex=True)
#delete nan records

raw_data = raw_data[raw_data['drugs_clean']!='No Response']


#lets select all of the relevant fields
raw_data = raw_data[['age','religion', 'offspring','education','job','drinks','smokes','drugs']]
#process_data[:100]



#this is an imbalanced class problem. will need to be careful when fitting
#never = 51804 recs
#sometimes = 7732
#often = 410

raw_data.job.value_counts()

#turn this into binary classificaiton. 
drug_map = {'never':0, 'sometimes':1, 'often':1}
process_data = pd.DataFrame()
process_data['drug_map']=raw_data.drugs.map(drug_map)

#replace all other nans
raw_data = raw_data.replace(np.nan,'',regex=True)

#map other columns
#religion
from sklearn import preprocessing
le_rel = preprocessing.LabelEncoder()
le_rel.fit(raw_data['religion'])
process_data['religion_map']=le_rel.transform(raw_data['religion'])
#offspring
le_off = preprocessing.LabelEncoder()
le_off.fit(raw_data['offspring'])
process_data['offspring_map'] = le_off.transform(raw_data['offspring'])
#education
le_edu = preprocessing.LabelEncoder()
le_edu.fit(raw_data['education'])
process_data['education_map'] = le_edu.transform(raw_data['education'])
#job
le_job = preprocessing.LabelEncoder()
le_job.fit(raw_data['job'])
process_data['job'] = le_job.transform(raw_data['job'])
#drinks
drink_map = {'not at all': 0, 'rarely': 1, 'socially': 2, 'often': 3, 'very often': 4, 'desperately': 5, '':6}
process_data['drink_map']=raw_data.drinks.map(drink_map)
#smokes
smoke_map = {'no':0, 'trying to quit':1,'when drinking':2, 'sometimes':3, 'yes':4, '':5}
process_data['smoke_map']=raw_data.smokes.map(smoke_map)

process_data['age'] = raw_data['age']


#split into X y
y = process_data.pop('drug_map')
X = process_data.values

#preprocessing
min_max = preprocessing.MinMaxScaler()
X_scale = min_max.fit_transform(X)

#run the model
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=.2)

##########################
#classifiers

#n_nearest neighbors
nbr_cnt = [3,4,5,6,7,8,9,10,11,12,13,14,15]
nbr_precision = []
nbr_recall = []
for n in nbr_cnt:
    nbr = neighbors.KNeighborsClassifier(n_neighbors=n, weights='distance')
    nbr.fit(x_train, y_train)
    nbr_predict = nbr.predict(x_test)
    #nbr_accuracy.append(nbr.score(x_test, y_test))
    nbr_precision.append(metrics.precision_score(y_test, nbr_predict))
    nbr_recall.append(metrics.recall_score(y_test, nbr_predict))

plt.plot(nbr_cnt, nbr_precision, label='precision')
plt.plot(nbr_cnt, nbr_recall, label='recall')
plt.legend()
plt.xlabel('N neighbors')
plt.ylabel('accuracy')
plt.title('Accuracy for N Neighbors - Class 1')
plt.show()

nbr = neighbors.KNeighborsClassifier(n_neighbors=9, weights='distance')
nbr.fit(x_train, y_train)
nbr_predict = nbr.predict(x_test)
print(nbr.score(x_test, y_test))
#precision and recall
print(metrics.classification_report(y_test, nbr_predict))


#Support Vector Machine
gammas = [1,0.5, 0.1, 0.05, 0.01,0.001]
precision = []
recall = []
for g in gammas:
    sv=SVC(class_weight='balanced', gamma=g, C=1)
    sv.fit(x_train, y_train)
    sv_predict=sv.predict(x_test)
    #accuracy.append(sv.score(x_test, y_test))
    precision.append(metrics.precision_score(y_test, sv_predict))
    recall.append(metrics.recall_score(y_test, sv_predict))


plt.plot(gammas, precision, label='precision')
plt.plot(gammas, recall, label='recall')
plt.legend()
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.title('Accuracy for Gamma Values - Class 1')
plt.show()



sv = SVC(class_weight='balanced', gamma=0.01, C=1)
sv.fit(x_train, y_train)
sv_predict = sv.predict(x_test)
print(sv.score(x_test, y_test))
print(metrics.classification_report(y_test, sv_predict))

###########################
#regressors

#regressors dont really make sense to use here since this is a binary problem
"""
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
lin_reg_predict=lin_reg.predict(x_test)
print(lin_reg.score(x_test, y_test))


nbr_reg = neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance')
nbr_reg.fit(x_train, y_train)
nbr_reg.score(x_test, y_test)

"""