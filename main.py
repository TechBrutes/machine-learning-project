import pandas   #pandas help handle the dataset. they frame the data nicely
from sklearn import preprocessing # to scale the numerical data
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import Perceptron
from sklearn.metrics import r2_score
import joblib # to export the models
import matplotlib.pyplot as plt

df = pandas.read_csv('assets/xAPI-Edu-Data.csv')

#PREPROCESSING THE DATA
#-----------------------
#Translate categorical to numerical data

#Class
class_dict = {
    "L" : 0,
    "M" : 1,
    "H" : 2,
}

#Replace the old values
df = df.replace({"Class" : class_dict})

#Scale all the numerical fields
df["raisedhands"] = preprocessing.scale(df["raisedhands"])
df["VisITedResources"] = preprocessing.scale(df["VisITedResources"])
df["AnnouncementsView"] = preprocessing.scale(df["AnnouncementsView"])
df["Discussion"] = preprocessing.scale(df["Discussion"])

#convert categorical fields into dummy data (to be represented numerically)
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
# print(dtree.feature_names_in_)
joblib.dump(dtree, 'models/dtree.py') # export decision tree model

plot_tree(dtree, max_depth=4, filled=True, feature_names=dtree.feature_names_in_)
plt.show()

#THE PERCEPTRON CLASSIFIER
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(train_x, train_y)

print("Accuracy of Perceptron classifier:\t", clf.score(test_x, test_y))
joblib.dump(clf, 'models/perceptron.py') # export Perceptron model

#THE SVM CLASSIFIER
from sklearn.svm import SVC

svm_clf = SVC(gamma='auto')
svm_clf.fit(train_x, train_y)
print("Accuracy of SVM classifier:\t\t", svm_clf.score(test_x, test_y))
joblib.dump(svm_clf, 'models/svm.py') # export svm model

#test run for loading model
loaded_model = joblib.load('models/svm.py')
result = loaded_model.score(test_x, test_y)
print('\t------------------')
print('Result from imported model (SVM):\t', result, '\n')
