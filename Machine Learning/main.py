#import thr 'numpy' and 'pandas' library for calculation, data manipulation and analysis.
import numpy as np
import pandas as pd

#import the RandomForestClassifier class from the sklearn (scikit-learn) library. 
from sklearn.ensemble import RandomForestClassifier

#import specific functions for evaluating the performance of the machine learning model, such as accuracy, classification report, and confusion matrix.
from sklearn.metrics import (accuracy_score, classification_report,confusion_matrix)

#import the 'train_test_split' function from scikit-learn
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/anish/OneDrive/Desktop/MINI_PROJECT_5TH_SEM/Dataset/Symptom.csv")
df.drop(['Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13',
'Symptom_14', 'Symptom_15', 'Symptom_16', 'Symptom_17'], axis=1, inplace=True)   #axis = 1 means work along the columns,
                                                                                #changes are happening inplace, i.e in same file
cols = df.columns
data = df[cols].values.flatten()   #make 1 d numpy array
s = pd.Series(data)    #create labeled pandas series
s = s.str.strip()   #remove whitespaces from strings in the series
s = s.values.reshape(df.shape)    #reshape as original dataframe
df = pd.DataFrame(s, columns=df.columns)   #replace the original dataframe with the modified dataframe
df = df.fillna(0)   #fill missing value with 0
vals = df.values     #numpy array

df2 = pd.read_csv("C:/Users/anish/OneDrive/Desktop/MINI_PROJECT_5TH_SEM/Dataset/Disease Specialist.csv")
specialist = df2['Specialist'].tolist()
edd = df2['Disease'].tolist()

df1 = pd.read_csv("C:/Users/anish/OneDrive/Desktop/MINI_PROJECT_5TH_SEM/Dataset/Symptom Severity.csv")
symptoms = df1['Symptom'].unique()
for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]    #replaces the matching symptoms in vals with their
                                                                                                     #weights in severity
            
d = pd.DataFrame(vals, columns=cols)
d = d.replace('dischromic _patches', 6)
d = d.replace('spotting_ urination', 6)
df = d.replace('foul_smell_of urine', 5)

data = df.iloc[:, 1:].values   #numpy array from  all rows and column index 1
labels = df['Disease'].values    #create a series of first column or disease names

# Data splitting for training and testing
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)   #20 percent for testing

#Logistic Regression
lr_classifier = LogisticRegression(multi_class = "ovr")
lr_classifier.fit(train_data, train_labels)
score = lr_classifier.score(test_data, test_labels)
print("The score for Logistic Regression is ",score*100,"%")

#SVM Model Training
svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    svc_classifier = SVC(kernel = kernels[i])
    svc_classifier.fit(train_data, train_labels)
    svc_scores.append(svc_classifier.score(test_data, test_labels))
for i in range (len(kernels)):
    svc_scores[i] = round(svc_scores[i], 2)
print("Accuracy of svm: ", svc_scores)

# Random Forest model training
rf = RandomForestClassifier(n_estimators=100)    #n_estimatoes gives the number of decision trees
rf = rf.fit(train_data, train_labels)    #training happens here

# Testing the model on the test set
predictions = rf.predict(test_data)

# Model evaluation
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy}")     #accuracy value inserted in the curly braces

psymp = ["itching", "vomiting", "nausea", "cough", "acidity"]
psymptoms = np.asarray(psymp)
a = np.array(df1["Symptom"])
b = np.array(df1["weight"])
for j in range(len(psymptoms)):
    for k in range(len(a)):
        if psymptoms[j] == a[k]:
            psymptoms[j] = b[k]
psy = [psymptoms]
prob = rf.predict(psy)
print(prob[0])
spec = prob[0]
for l in range(len(edd)):
    if spec == edd[l]:
        print(specialist[l])
