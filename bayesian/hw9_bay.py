###
### CS667 Data Science with Python, Homework 8, Jon Organ
###

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("cmg_weeks.csv")

# Question 1 ========================================================================================
print("Question 1:")

X = df[['Avg_Return', 'Volatility']][df['Week'] <= 50].values
Y = df['Color'][df['Week'] <= 50].values

NB_classifier = GaussianNB().fit(X, Y)

prediction = NB_classifier.predict(df[['Avg_Return', 'Volatility']][(df['Week'] > 50) & (df['Week'] <= 100)].values)
actual = df['Color'][(df['Week'] > 50) & (df['Week'] <= 100)].values


score = NB_classifier.score(df[['Avg_Return', 'Volatility']][(df['Week'] > 50) & (df['Week'] <= 100)].values, 
	df['Color'][(df['Week'] > 50) & (df['Week'] <= 100)].values)
print("Gaussian Naive Bayesian year 2 accuracy: " + str(score * 100) + "%")


print("\n")
# Question 2 ========================================================================================
print("Question 2:")