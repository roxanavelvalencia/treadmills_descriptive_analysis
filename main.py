# Load the necessary packages
# import numpy as np
import pandas as pd

# Load function from sklean
from sklearn import linear_model

# Load the Cardio Dataset
mydata = pd.read_csv('./data/CardioGoodFitness.csv')

print('Describe', mydata.describe(include='all'))
print('Info', mydata.info())

# Gender vs Product
print(pd.crosstab(mydata['Product'], mydata['Gender']))
# Product vs Marital Status
print(pd.crosstab(mydata['Product'], mydata['MaritalStatus']))

# Compare Product/Gender/Income/Miles related to Martial Status
print(pd.pivot_table(mydata, index=['Product', 'Gender'], columns=['MaritalStatus'], aggfunc=len))
print(pd.pivot_table(mydata, 'Income', index=['Product', 'Gender'], columns=['MaritalStatus']))
print(pd.pivot_table(mydata, 'Miles', index=['Product', 'Gender'], columns=['MaritalStatus']))

# Mean & Std Dev of Age
print('Age Std Dev', mydata['Age'].std())
print('Age Mean', mydata['Age'].mean())

# Create linear regression object
regression = linear_model.LinearRegression()

y = mydata['Miles']
x = mydata[['Usage', 'Fitness']]

# Train the model using the training sets
regression.fit(x, y)

print('Coefficient', regression.coef_)
print('Interception', regression.intercept_)
