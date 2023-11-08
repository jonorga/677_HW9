###
### CS667 Data Science with Python, Homework 8, Jon Organ
###

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


df = pd.read_csv("cmg_weeks.csv")

# Question 1 ========================================================================================
print("Question 1:")

X = df[['Avg_Return', 'Volatility']][df['Week'] <= 50].values
Y = df['Color'][df['Week'] <= 50].values

LDA = LinearDiscriminantAnalysis()
QDA = QuadraticDiscriminantAnalysis()
LDA.fit(X, Y)
QDA.fit(X, Y)

print("Linear Discriminant Analysis equation: y = " + str(LDA.coef_[0,0]) + "x + " + str(LDA.intercept_[0]))
print("Quadratic Discriminant Analysis returned the following means: " + str(QDA.means_))

print("\n")
# Question 2 ========================================================================================
print("Question 2:")


X_test = df[['Avg_Return', 'Volatility']][(df['Week'] > 50) & (df['Week'] <= 100)].values
Y_test = df['Color'][(df['Week'] > 50) & (df['Week'] <= 100)].values
LDA_score = LDA.score(X_test, Y_test)
QDA_score = QDA.score(X_test, Y_test)
if LDA_score > QDA_score:
	print("LDA is more accurate.")
else:
	print("QDA is more accurate.")
print("LDA score: " + str(LDA_score))
print("QDA score: " + str(QDA_score))


print("\n")
# Question 3 ========================================================================================
print("Question 3:")




