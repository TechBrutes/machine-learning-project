import pandas   #pandas help handle the dataset. they frame the data nicely
import matplotlib.pyplot as plt #this helps plot data visualizations

df = pandas.read_csv('xAPI-Edu-Data.csv')
print(df.corr)

df['Discussion'].plot(kind = 'hist')
plt.show()