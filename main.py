import pandas   #pandas help handle the dataset. they frame the data nicely
import matplotlib.pyplot as plt #this helps plot data visualizations
import seaborn #a nice addition to matplotlib
from sklearn import preprocessing #to scale the numerical data

df = pandas.read_csv('xAPI-Edu-Data.csv')
print(df.corr)


#plotting visualizations for each feature in the dataset

"""
#CATEGORICAL DATA
# Class (the target feature)
seaborn.countplot(x='Class', data=df, order=['L', 'M', 'H'])
plt.title("Count plot for Class (target feature")
plt.show()

#gender
fig, axarr  = plt.subplots(2) #to plot 2 graphs, one more detailed
seaborn.countplot(x='gender', data=df, order=['M', 'F'], ax=axarr[0])
seaborn.countplot(x='gender', hue='Class', data=df, order=['M', 'F'], hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.title("Count plot for Gender")
plt.show()

#nationality
fig, axarr  = plt.subplots(2)
seaborn.countplot(x='NationalITy', data=df, ax=axarr[0])
seaborn.countplot(x='NationalITy', data=df, hue='Class', hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.suptitle("Count plot for Nationality")
plt.show()


#place of birth
fig, axarr  = plt.subplots(2)
seaborn.countplot(x='PlaceofBirth', data=df, ax=axarr[0])
seaborn.countplot(x='PlaceofBirth', data=df, hue='Class', hue_order = ['L', 'M', 'H'], ax=axarr[1])
plt.suptitle("Count plot for Place of birth")
plt.show()

#Stage Id
fig, axarr = plt.subplots(2)
seaborn.countplot(x='StageID', data=df, ax=axarr[0])
seaborn.countplot(x='StageID', data=df, ax=axarr[1], hue='Class', hue_order=['L','M','H'])
plt.suptitle("Count plot for Stage ID")
plt.show()

#grade Id
fig, axarr = plt.subplots(2)
seaborn.countplot(x='GradeID', data=df, ax=axarr[0])
seaborn.countplot(x='GradeID', data=df, ax=axarr[1], hue='Class', hue_order=['L','M','H'])
plt.suptitle("Count plot for Grade ID")
plt.show()

#section Id
fig, axarr = plt.subplots(2)
seaborn.countplot(x='SectionID', data=df, ax=axarr[0])
seaborn.countplot(x='SectionID', data=df, ax=axarr[1], hue='Class', hue_order=['L','M','H'])
plt.suptitle("Count plot for Section ID")
plt.show()

#Topic
fig, axarr = plt.subplots(2)
seaborn.countplot(x='Topic', data=df, ax=axarr[0])
seaborn.countplot(x='Topic', data=df, ax=axarr[1], hue='Class', hue_order=['L','M','H'])
plt.suptitle("Count plot for Topic")
plt.show()

#Semester
fig, axarr = plt.subplots(2)
seaborn.countplot(x='Semester', data=df, ax=axarr[0])
seaborn.countplot(x='Semester', data=df, ax=axarr[1], hue='Class', hue_order=['L','M','H'])
plt.suptitle("Count plot for Semester")
plt.show()

#Relation
fig, axarr = plt.subplots(2)
seaborn.countplot(x='Relation', data=df, ax=axarr[0])
seaborn.countplot(x='Relation', data=df, ax=axarr[1], hue='Class', hue_order=['L','M','H'])
plt.suptitle("Count plot for Relation")
plt.show()

#ParentAnsweringSurvey
fig, axarr = plt.subplots(2)
seaborn.countplot(x='ParentAnsweringSurvey', data=df, ax=axarr[0])
seaborn.countplot(x='ParentAnsweringSurvey', data=df, ax=axarr[1], hue='Class', hue_order=['L','M','H'])
plt.suptitle("Count plot for ParentAnsweringSurvey")
plt.show()

#ParentschoolSatisfaction
fig, axarr = plt.subplots(2)
seaborn.countplot(x='ParentschoolSatisfaction', data=df, ax=axarr[0])
seaborn.countplot(x='ParentschoolSatisfaction', data=df, ax=axarr[1], hue='Class', hue_order=['L','M','H'])
plt.suptitle("Count plot for ParentschoolSatisfaction")
plt.show()

#StudentAbsenceDays
fig, axarr = plt.subplots(2)
seaborn.countplot(x='StudentAbsenceDays', data=df, ax=axarr[0])
seaborn.countplot(x='StudentAbsenceDays', data=df, ax=axarr[1], hue='Class', hue_order=['L','M','H'])
plt.suptitle("Count plot for StudentAbsenceDays")
plt.show()

#NUMERICAL DATA
#pairplots help us visualize how numeric fields are scattered against each other
#the class hue gives the plot meaning. we can know where good performers and poor performers lie in the scatter diagrams
seaborn.pairplot(df, hue="Class", diag_kind="kde", hue_order=['L','M','H'])
plt.suptitle('Pairplot for all the numerical fields in the dataset')
plt.show()

"""

#PREPROCESSING THE DATA
#-----------------------
#Translate GradeId from categorical to numerical data
gradeID_dict = {
    "G-01" : 1,
    "G-02" : 2,
    "G-03" : 3,
    "G-04" : 4,
    "G-05" : 5,
    "G-06" : 6,
    "G-07" : 7,
    "G-08" : 8,
    "G-09" : 9,
    "G-10" : 10,
    "G-11" : 11,
    "G-12" : 12,
}

#Now replace the old values with the new values
df = df.replace({"GradeID" : gradeID_dict})

#Translate Class from categorical to numerical data also
class_dict = {
    "L" : -1,
    "M" : 0,
    "H" : 1,
}

#Then replace the old values
df = df.replace({"Class" : class_dict})

#Scale all the numerical fields

scaled_GradeID = preprocessing.scale(df["GradeID"])
scaled_raisedHands = preprocessing.scale(df["raisedhands"])
scaled_visitedResources = preprocessing.scale(df["VisITedResources"])
scaled_announcementsView = preprocessing.scale(df["AnnouncementsView"])
scaled_discussion = preprocessing.scale(df["Discussion"])

#convert categorical fields into dummy data (which is the way categorical data can be represented numerically, for the algorithm to understand)
df = pandas.get_dummies(df, columns = [
    "gender", "NationalITy", "PlaceofBirth", "SectionID", 
    "StageID","Topic","Semester","Relation","ParentAnsweringSurvey",
    "ParentschoolSatisfaction","StudentAbsenceDays"
])

print(df.head())

#CONFIGURING AND TRAINING THE MODELS