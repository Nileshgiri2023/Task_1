# Task_1
Task 1 (The Sparks Foundation Internship Programme):Prediction using Supervised Machine Learning
Author:NILESH GIRI
TASK 1: PREDICTION USING SUPERVISED MACHINE LEARNING
#Importing all libraries required in this task
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
​
#Reading data from give link
#url_task1 = "http://bit.ly/w-data" #tHE GIVEN LINK IS NOT WORKING
data = pd.read_csv("C:\\Users\\user\\OneDrive\\Desktop\\TSF PROJECT\\task 1.csv")
print("data imported successfully")


data.head(26)


#Reading data from give link
#url_task1 = "http://bit.ly/w-data" #tHE GIVEN LINK IS NOT WORKING
data = pd.read_csv("C:\\Users\\user\\OneDrive\\Desktop\\TSF PROJECT\\task 1.csv")
print("data imported successfully")
​
​
data.head(26)
​
​
data.shape
​
data.describe() #Gives mean,std ,min,max
data.info()
#Plotting the dataset
data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage Scores')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.show()
#Preparing the data
x = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  
x
y
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x,y)
prediction = model.predict([[2.5],[5.8]])
prediction
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
print("Training complete")
#Plotting the regression line
line = regressor.coef_*x+regressor.intercept_
​
#plotting the test data 
plt.scatter(x,y)
plt.plot(x,line)
plt.show()
regressor.intercept_
​
#Now we have trained our algorithm ,it's time to make some predictions
print(x_test)
y_pred = regressor.predict(x_test) # Predicting the Scores
y_pred
#comparing actual Vs Predicted Scores
df =pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df
#Now We test for the Given data
hours = 9.25
print(hours)
own_pred = regressor.predict([[hours]])
print("No of Hours ={}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))
#Evaluating the model
#The final step is to evaluate the performance of algorithm.This step is 
#particularly important to compare how well different algorithms perform on a particular datasets.
#For simplicity here ,we have chosen the mean square error.There are many such metrics.
from sklearn import metrics 
print('Mean Absolute Error: ',metrics.mean_absolute_error(y_test,y_pred))

#Evaluating the model
#The final step is to evaluate the performance of algorithm.This step is 
#particularly important to compare how well different algorithms perform on a particular datasets.
#For simplicity here ,we have chosen the mean square error.There are many such metrics.
from sklearn import metrics 
print('Mean Absolute Error: ',metrics.mean_absolute_error(y_test,y_pred))
​
