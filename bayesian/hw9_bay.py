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

score = NB_classifier.score(df[['Avg_Return', 'Volatility']][(df['Week'] > 50) & (df['Week'] <= 100)].values, 
	df['Color'][(df['Week'] > 50) & (df['Week'] <= 100)].values)
print("Gaussian Naive Bayesian year 2 accuracy: " + str(score * 100) + "%")


print("\n")
# Question 2 ========================================================================================
print("Question 2:")
print("Confusion matrix:")
prediction = NB_classifier.predict(df[['Avg_Return', 'Volatility']][(df['Week'] > 50) & (df['Week'] <= 100)].values)
actual = df['Color'][(df['Week'] > 50) & (df['Week'] <= 100)].values

cm_df = pd.DataFrame()
cm_df['Actual'] = actual
cm_df['Predicted'] = prediction
confusion_matrix = pd.crosstab(cm_df['Actual'], cm_df['Predicted'])
print(confusion_matrix)


print("\n")
# Question 3 ========================================================================================
print("Question 3:")




