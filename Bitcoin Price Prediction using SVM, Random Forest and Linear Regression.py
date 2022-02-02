import pandas as pd
import numpy as np
import math
import sklearn 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt 
import seaborn as sb 
from sklearn import preprocessing

data=pd.read_csv("C:\\Users\\user\\OneDrive\\Desktop\\Bitcon price prediction\\bitcoin_data.csv", parse_dates=["Timestamp"], index_col="Timestamp")

data=data.drop(columns=['Volume (BTC)','Volume (Currency)'])

data.head(10)
data.tail(10)
data.shape
data.isnull().values.any()
data.isnull().sum()
data.info()
data.describe()
data.cov()
data.corr()

plt.figure(figsize=(9,5))
data.plot(kind="hist")

corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,8))
sb.heatmap(corrmat,ax=ax,cmap="YlGnBu",linewidths=0.1)

corrmat = data.corr() 
cg = sb.clustermap(corrmat, cmap ="YlGnBu", linewidths = 0.1); 
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0)
cg

corrmat = data.cov() 
f, ax = plt.subplots(figsize =(9, 8)) 
sb.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)

corrmat = data.cov() 
cg = sb.clustermap(corrmat, cmap ="YlGnBu", linewidths = 0.1); 
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0) 
cg 

plt.scatter(data.corr(),data.cov())

data.hist()
plt.show()

sb.pairplot(data)

# A variable for predicting 30 days out into the future
forecast_out = 30
#Create another column (the target or dependent variable) shifted 30 units up
data['Prediction'] = data[['Weighted_Price']].shift(-forecast_out)

#print the new data set
print(data.tail(31))
print(data.head(10))

# Create the independent data set (x)
# Convert the dataframe to a numpy array
x = np.array(data.drop(['Prediction'],1))

#Remove the last 30 rows
x = x[:-forecast_out]
print(x)

# Create the dependent data set (y)
# Convert the dataframe to a numpy array (All of the values including the NaN's)
y = np.array(data['Prediction'])
# Get all of the y values except the last 30 rows
y = y[:-forecast_out]
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from sklearn.model_selection import KFold
kf=KFold(n_splits=10)
Kf

for train_index, test_index in kf.split(data):
print(train_index, test_index)

print(x_train)
print(x_test)
print(y_train)
print(y_test)
x_train.shape
x_test.shape
y_train.shape
y_test.shape

#set x_forecast equal to the last 30 rows of the original data set from the Weighted_price
x_forecast = np.array(data.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)

# Create and train the Support Vector Machine
from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=1e3, gamma='scale')
svr_rbf.fit(x_train, y_train)

# The best possible score is: 1.0
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)

svm_prediction = svr_rbf.predict(x_test)
# Calculate: the absolute errors
errors = abs(svm_prediction - y_test)
# Mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))


# Calculate: mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

#predicted values
svm_prediction = svr_rbf.predict(x_test)
print(svm_prediction)
print("------------------------------------")
#actual values
print(y_test)

#The model prediction for the next 30 days
svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)

#The actual price for the next 30 days
data.tail(forecast_out)

from sklearn.linear_model import LinearRegression
# Create and train the Linear Regression  Model
lr = LinearRegression()
# Train the model
lr.fit(x_train, y_train)

# The best possible score is 1.0
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)


lr_prediction = lr.predict(x_test)
# Calculate: the absolute errors
errors = abs(lr_prediction - y_test)
# Mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))


# Calculate: mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

#predicted values
lr_prediction = lr.predict(x_test)
print(lr_prediction)
print("-----------------------")
#actual values
print(y_test)

# Print linear regression model predictions for the next 30 days
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)

data.tail(forecast_out)

from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(x_train, y_train)

rf_confidence = lr.score(x_test, y_test)
print("rf confidence: ", rf_confidence)

# Use the forest's predict method on the test data
rf_predictions = rf.predict(x_test)
# Calculate: the absolute errors
errors = abs(rf_predictions - y_test)
# Mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))

# Calculate: mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

#predicted values
rf_prediction = rf.predict(x_test)
print(rf_prediction)
print("-----------------------")
#actual values
print(y_test)

# Print linear regression model predictions for the next 30 days
rf_prediction = rf.predict(x_forecast)
print(rf_prediction)

data.tail(forecast_out)
