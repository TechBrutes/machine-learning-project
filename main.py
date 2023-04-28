import pandas   #pandas help handle the dataset. they frame the data nicely
import matplotlib.pyplot as plt #this helps plot data visualizations
import seaborn #a nice addition to matplotlib

df = pandas.read_csv('xAPI-Edu-Data.csv')
print(df.corr)

#plot visualizations for each feature in the dataset
#firstly, plot the count of the target feature (the one we're trying to predict), 'Class'
seaborn.countplot(x='Class', data=df, order=['L', 'M', 'H'])
plt.title("Count plot for Class (target feature")
plt.show()

#gender
seaborn.countplot(x='gender', data=df, order=['M', 'F'])
plt.title("Count plot for Gender")
plt.show()

#nationality
seaborn.countplot(x='NationalITy', data=df)
plt.title("Count plot for Nationality")
plt.show()

#place of birth
seaborn.countplot(x='PlaceofBirth', data=df)
plt.title("Count plot for Place of birth")
plt.show()


#Stage Id
seaborn.countplot(x='StageID', data=df)
plt.title("Count plot for Stage ID")
plt.show()


#grade Id
seaborn.countplot(x='GradeID', data=df)
plt.title("Count plot for Grade ID")
plt.show()