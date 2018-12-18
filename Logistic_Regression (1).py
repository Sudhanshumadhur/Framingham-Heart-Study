import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Loading the dataframe using pandas
data = pd.read_csv('Desktop\Train_data.csv')

# Dataframe dimensions
rows, columns = data.shape

row=['sex','age','education','currentSmoker','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp',
	 'diabetes','totChol','BMI','heartRate','glucose','sysBP','diaBP']
#print len(row)

# Filling the missing values 

data['sex'] = data['sex'].fillna(0)

minimum_education = min(data['education'])
data['education'] = data['education'].fillna(minimum_education)

average_age = sum(data['age'])/rows
data['age'] = data['age'].fillna(average_age)

data['currentSmoker'] = data['currentSmoker'].fillna(0)

data['cigsPerDay'] = data['cigsPerDay'].fillna(0)

data['BPMeds'] = data['BPMeds'].fillna(0)

data['prevalentStroke'] = data['prevalentStroke'].fillna(0)

data['prevalentHyp'] = data['prevalentHyp'].fillna(0)

data['diabetes'] = data['diabetes'].fillna(0)

average_chol = sum(data['totChol'])/rows
data['totChol'] = data['totChol'].fillna(average_chol)

average_BMI = sum(data['BMI'])/rows
data['BMI'] = data['BMI'].fillna(25)

average_heartRate = sum(data['heartRate'])/rows
data['heartRate'] = data['heartRate'].fillna(average_heartRate)

average_glucose = sum(data['glucose'])/rows
data['glucose'] = data['glucose'].fillna(70)

data['sysBP'] = data['sysBP'].fillna(0)
data['diaBP'] = data['diaBP'].fillna(0)

for i in range(rows):
	if data['sysBP'][i] == 0 and data['diaBP'] == 0:
		data['sysBP'][i] = 120
		data['diaBP'][i] = 80
	elif data['sysBP'][i] == 0:
		data['sysBP'][i] = data['diaBP'][i] + 40
	elif data['diaBP'][i] == 0:
		data['diaBP'][i] = data['sysBP'] - 40
data.to_csv('Framingham2.csv')

data['TenYearCHD'] = data['TenYearCHD'].fillna(0)

# data = data.fillna(0)

# Defining the target variable
y = data['TenYearCHD']

# Deleting the target variable from the dataframe
del data['TenYearCHD']

# Defining the features vector
X = data

# Test-Train split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1)


# Applying the model onto the dataset
LogisticR = LogisticRegression(solver = 'liblinear')

X_train=np.array(X_train)
y_train=np.array(y_train)

X_train[np.isnan(X_train)] = np.median(X_train[~np.isnan(X_train)])

LogisticR.fit(X_train,y_train)
print "coefficients",LogisticR.coef_
print "intercept",LogisticR.intercept_
print 'Training accuracy :', LogisticR.score(X_train,y_train)
X_test=np.array(X_test)
X_test[np.isnan(X_test)] = np.median(X_test[~np.isnan(X_test)])
y_test=np.array(y_test)

y_pred = LogisticR.predict(X_test)
print 'Test accuracy :', accuracy_score(y_test,y_pred, normalize = True)
#print y_pred,y_test

data = data.fillna(0)
print LogisticR.get_params(deep=False)
# Constructing a covariance matrix 
covariance_matrix = data.cov()

covariance_matrix.to_csv('covariance_matrix.csv')

# Constructing a correlation coefficient matrix - pearson correlation coefficient
correlation_matrix = data.corr(method = 'pearson')
correlation_matrix.to_csv('correlation_matrix.csv')



