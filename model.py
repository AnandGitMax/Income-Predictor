import os


#Import modules and libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump, load

#Read data
MyData=pd.read_csv("Income_Expense_Data.csv")

#Check data size
MyData.shape

#Check for top 5 rows
MyData.head()

#Check for null values
MyData.isnull().sum()

#Replace null values with median
MyData["Expense"].fillna((MyData["Expense"].median()), inplace=True)

#Check for outliers from data description
MyData.describe()

#Check for different percentiles to understand data better
pd.DataFrame(MyData["Age"]).describe(percentiles=(1,0.99,0.9,0.74,0.5,0.3,0.1,0.01))

#Visualize data by boxplot for Age column
plt.boxplot(MyData["Age"])
plt.show()


#Check outlier by definition and treating outliers
#Find median Age
Age_col_df=pd.DataFrame(MyData['Age'])
Age_median=Age_col_df.median()
#Find IQR(Inter quantile range) of Age column
Q3=Age_col_df.quantile(q=0.75)
Q1=Age_col_df.quantile(q=0.25)
IQR=Q3-Q1
#Derive boundaries of outliers
IQR_LL=int(Q1-1.5*IQR)
IQR_UL=int(Q3-1.5*IQR)
#Find and treat outliers for both lower and upper end
#Go to MyData and locate where the value is greater than IQR_UL, if you find those values and then replace it with 99 percentile of the data(Age).
MyData.loc[MyData['Age']>IQR_UL, 'Age']=int(Age_col_df.quantile(q=0.99))
MyData.loc[MyData['Age']>IQR_UL, 'Age']=int(Age_col_df.quantile(q=0.01))
#Check max age now
max(MyData['Age'])
#Check for different percentiles(after treatment)
pd.DataFrame(MyData["Age"]).describe(percentiles=(1,0.99,0.9,0.74,0.5,0.3,0.1,0.01))
#Check boxplot for Age column(after treatment)
plt.boxplot(MyData["Age"])
plt.show()

#Find coorelation between columns (data analysis)
#Find coorelation between Expense and Income (how Expense is varying with income)
x=MyData["Expense"]
y=MyData["Income"]

plt.scatter(x,y,label='Expense Vs Income')

#Find coorelation between Income and Age (how Income is varying with Age)
x=MyData["Age"]
y=MyData["Income"]

plt.scatter(x,y,label='Income Vs Age')

#Check the correlation matrix - to check the strength between 2 variabls
correlation_matrix=MyData.corr().round(2) #Give me the correlation of the values
f,ax=plt.subplots(figsize=(8,4))
sns.heatmap(data=correlation_matrix, annot=True)

#Separating features and response
features = ["Expense", "Age"]
response = ["Income"]
x=MyData[features].to_numpy()
y=MyData[response].to_numpy()

#Dividing data in test and train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Fitting the linear regression model
from sklearn import metrics
model = LinearRegression()
model.fit(x_train, y_train)

#Checking accuracy on test data
accuracy = model.score(x_test, y_test)
print(accuracy*100, '%')

# #Dumping the model Object
# import pickle
# pickle.dump(model, open('model.pkl','wb'))


dump(model, 'model.joblib')

#Reloading the model object
#model=pickle.load(open('model.pkl','rb'))

model = load('model.joblib')
print(model.predict([[10000,30]]))



