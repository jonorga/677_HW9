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

NB_point5 = GaussianNB(var_smoothing=0.5).fit(X, Y)
NB_1 = GaussianNB(var_smoothing=1).fit(X, Y)
NB_5 = GaussianNB(var_smoothing=5).fit(X, Y)

def Q1(NB, degree):
	score = NB.score(df[['Avg_Return', 'Volatility']][(df['Week'] > 50) & (df['Week'] <= 100)].values, 
		df['Color'][(df['Week'] > 50) & (df['Week'] <= 100)].values)
	print("Gaussian Naive Bayesian, df = " + str(degree) + ", year 2 accuracy: " + str(score * 100) + "%")

Q1(NB_point5, "0.5")
Q1(NB_1, "1")
Q1(NB_5, "5")


print("\n")
# Question 2 ========================================================================================
print("Question 2:")
print("Confusion matrices:")

def Q2(NB, degree):
	prediction = NB.predict(df[['Avg_Return', 'Volatility']][(df['Week'] > 50) & (df['Week'] <= 100)].values)
	actual = df['Color'][(df['Week'] > 50) & (df['Week'] <= 100)].values

	cm_df = pd.DataFrame()
	cm_df['Actual'] = actual
	cm_df['Predicted'] = prediction
	confusion_matrix = pd.crosstab(cm_df['Actual'], cm_df['Predicted'])
	print(degree, "degree confusion matrix")
	print(confusion_matrix, "\n")
	return confusion_matrix, cm_df

cm_point5, cmdf_point5 = Q2(NB_point5, "0.5")
cm_1, cmdf_1 = Q2(NB_1, "1")
cm_5, cmdf_5 = Q2(NB_5, "5")


print("\n")
# Question 3 ========================================================================================
print("Question 3:")

def Q3(confusion_matrix, degree):
	TP = confusion_matrix['Green'].iloc[0]
	TN = confusion_matrix['Red'].iloc[1]
	FP = confusion_matrix['Green'].iloc[1]
	FN = confusion_matrix['Red'].iloc[0]

	TPR = round((TP / (TP + FN) * 100), 2)
	TNR = round((TN / (TN + FP) * 100), 2)
	print(degree, "degree TPR and TNR")
	print("True positive rate: " + str(TPR) + "%")
	print("True negative rate: " + str(TNR) + "%\n")

Q3(cm_point5, "0.5")
Q3(cm_1, "1")
Q3(cm_5, "5")


print("\n")
# Question 4 ========================================================================================
print("Question 4:")
print("The best value of df is 0.5, it is 2% more accurate than normal Naive Bayesian")


print("\n")
# Question 5 ========================================================================================
print("Question 5:")
print("Buy-and-hold retrieved from previous assignment...")


balance = 100
file_len = len(cmdf_point5.index)
i = 0
while i < file_len:
	today_stock = balance / df['Close'].iloc[i]
	tmr_stock = balance / df['Close'].iloc[i + 1]
	difference = abs(today_stock - tmr_stock)
	if cmdf_point5['Actual'].iloc[i] == cmdf_point5['Predicted'].iloc[i]:
		balance += difference * df["Close"].iloc[i + 1]
	else:
		balance -= difference * df["Close"].iloc[i + 1]
	i += 1
final_balance = round(balance, 2)
if balance > 125.7:
	print("Gaussian Naive Bayes with df = 0.5 ($" + str(final_balance) 
		+ ") was more effective than Buy-and-hold ($125.70) for year 2")
else:
	print("Gaussian Naive Bayes with df = 0.5 ($" + str(final_balance) 
		+ ") was less effective than Buy-and-hold ($125.70) for year 2")

