###
### CS667 Data Science with Python, Homework 8, Jon Organ
###

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("cmg_weeks.csv")

# Question 1 ========================================================================================
print("Question 1:")

X = df[['Avg_Return', 'Volatility']][df['Week'] <= 50].values
Y = df['Color'][df['Week'] <= 50].values

def Q1(degree):
	NB_classifier = GaussianNB(var_smoothing=degree)
	NB_classifier.fit(X, Y)
	score = NB_classifier.score(df[['Avg_Return', 'Volatility']][(df['Week'] > 50) & (df['Week'] <= 100)].values, 
		df['Color'][(df['Week'] > 50) & (df['Week'] <= 100)].values)
	print("Gaussian Naive Bayesian, df = " + str(degree) + ", year 2 accuracy: " + str(score * 100) + "%")

Q1(0.5)
Q1(1)
Q1(5)


print("\n")
# Question 2 ========================================================================================
print("Question 2:")
print("Confusion matrices:")


