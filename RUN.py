import joblib
import pandas
from sklearn import preprocessing # to scale the new data

file_location = input("\nEnter the file address (.csv file format): ")

df = pandas.read_csv(file_location) #read the new data
X = pandas.read_csv(file_location)

#Scale all the numerical fields
X["raisedhands"] = preprocessing.scale(X["raisedhands"])
X["VisITedResources"] = preprocessing.scale(X["VisITedResources"])
X["AnnouncementsView"] = preprocessing.scale(X["AnnouncementsView"])
X["Discussion"] = preprocessing.scale(X["Discussion"])

#convert categorical fields into dummy data (to be represented numerically)
X = pandas.get_dummies(X, columns = [
    "NationalITy", "PlaceofBirth", "Topic", "GradeID", "gender", "StageID", "SectionID", "Semester", "Relation",
    "ParentAnsweringSurvey", "ParentschoolSatisfaction", "StudentAbsenceDays"
])

#import the decision tree model
loaded_model = joblib.load('models/dtree.py')
y = loaded_model.predict(X)

class_dict = {
    0 : "L",
    1 : "M",
    2 : "H"
}

# map numeric predictions back to understandable values for the user
y = [class_dict.get(x, x) for x in y]

#print the predictions next to the source data
df['Predicted Performance'] = y

print('\nBelow is your students\' data, and the corresponding predictions\n')
print(df)
print('\nThe above data has been written onto a file named \'Student performance predictions.csv\'')
df.to_csv('Student performance predictions.csv', index=False)