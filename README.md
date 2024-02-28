# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph. 
5. .Predict the regression for marks by using the representation of the graph. 
6. Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:

## Program to implement the simple linear regression model for predicting the marks scored.
## Developed by: YUVARAJ JOSHITHA
## RegisterNumber:21223240189  

### IMPORT REQUIRED PACKAGE
```py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('/content/student_scores (2).csv')
print(dataset)
```
### READ CSV FILES
```py
dataset=pd.read_csv('/content/student_scores (2).csv')
print(dataset.head())
print(dataset.tail())
```
### COMPARE DATASET
```py
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
```
### PRINT PREDICTED VALUE
```py
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
```
### GRAPH PLOT FOR TRAINING SET
```py
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
### GRAPH PLOT FOR TESTING SET
```py
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
### PRINT THE ERROR
```py
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```

## Output:
### To Read All CSV Files
![Screenshot 2024-02-28 161032](https://github.com/Joshitha-YUVARAJ/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742770/11a3bf41-ebe2-4a5f-962e-73fe6029b6be)


### To Read Head and Tail Files

![Screenshot 2024-02-28 161052](https://github.com/Joshitha-YUVARAJ/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742770/93aa06cc-7f33-4f97-90a4-beac2bfc2401)


### Compare Dataset

![Screenshot 2024-02-28 161109](https://github.com/Joshitha-YUVARAJ/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742770/895e6de9-751b-4f9e-b801-abb6da08e267)



### Predicted Value
![Screenshot 2024-02-28 161121](https://github.com/Joshitha-YUVARAJ/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742770/422b3fd9-5a23-43c3-b015-3c5cc8ecd537)



### Graph For Training Set

![Screenshot 2024-02-28 161154](https://github.com/Joshitha-YUVARAJ/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742770/e80589e0-d967-4a28-87fe-cef81e166159)


### Graph For Testing Set
![Screenshot 2024-02-28 161220](https://github.com/Joshitha-YUVARAJ/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742770/ee2a1bb4-eaff-4291-8cbd-b6d7043ba5d5)


### Error
![Screenshot 2024-02-28 161231](https://github.com/Joshitha-YUVARAJ/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742770/6bbe22aa-cf71-4884-800d-8050395dabad)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.


