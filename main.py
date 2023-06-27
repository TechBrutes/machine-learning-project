import pandas   #pandas help handle the dataset. they frame the data nicely
from sklearn import preprocessing # to scale the numerical data
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pandas.read_csv('xAPI-Edu-Data.csv')
print(df.corr)


#PREPROCESSING THE DATA
#-----------------------
#Translate categorical to numerical data

#Class
class_dict = {
    "L" : 0,
    "M" : 1,
    "H" : 2,
}

#Then replace the old values
df = df.replace({"Class" : class_dict})


#Scale all the numerical fields
df["raisedhands"] = preprocessing.scale(df["raisedhands"])
df["VisITedResources"] = preprocessing.scale(df["VisITedResources"])
df["AnnouncementsView"] = preprocessing.scale(df["AnnouncementsView"])
df["Discussion"] = preprocessing.scale(df["Discussion"])

print(df)
#convert categorical fields into dummy data (which is the way categorical data can be represented numerically, for the algorithm to understand)
df = pandas.get_dummies(df, columns = [
    "NationalITy", "PlaceofBirth", "Topic", "GradeID", "gender", "StageID", "SectionID", "Semester", "Relation",
    "ParentAnsweringSurvey", "ParentschoolSatisfaction", "StudentAbsenceDays"
])

#CONFIGURING AND TRAINING THE MODELS

# First split the training and testing data

train_set = df.sample(frac = 0.8) # 80% of the dataset for training
test_set = df.loc[~df.index.isin(train_set.index)] #and 20% for testing

train_x = train_set.loc[:, lambda x: [l for l in df if l != "Class"]]
train_y = train_set["Class"]

test_x = test_set.loc[:, lambda x: [l for l in df if l != "Class"]]
test_y = test_set["Class"]

# NOW TRAIN THE THE DECISION TREE CLASSIFIER
dtree = DecisionTreeClassifier()
dtree = dtree.fit(train_x, train_y)

print("Accuracy of Decision tree classifier:\t", dtree.score(test_x, test_y))


#THE PERCEPTRON CLASSIFIER
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(train_x, train_y)
Perceptron()
print("Accuracy of Perceptron classifier:\t", clf.score(train_x, train_y))

#THE SVM CLASSIFIER
from sklearn.svm import SVC

svm_clf = SVC(gamma='auto')
svm_clf.fit(train_x, train_y)
print("Accuracy of SVM classifier:\t\t", svm_clf.score(test_x, test_y))