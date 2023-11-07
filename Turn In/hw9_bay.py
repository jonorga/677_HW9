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

TP = confusion_matrix['Green'].iloc[0]
TN = confusion_matrix['Red'].iloc[1]
FP = confusion_matrix['Green'].iloc[1]
FN = confusion_matrix['Red'].iloc[0]

TPR = round((TP / (TP + FN) * 100), 2)
TNR = round((TN / (TN + FP) * 100), 2)
print("True positive rate: " + str(TPR) + "%")
print("True negative rate: " + str(TNR) + "%")



print("\n")
# Question 4 ========================================================================================
print("Question 4:")
print("Buy-and-hold retrieved from previous assignment...")


balance = 100
file_len = len(cm_df.index)
i = 0
while i < file_len:
	today_stock = balance / df['Close'].iloc[i]
	tmr_stock = balance / df['Close'].iloc[i + 1]
	difference = abs(today_stock - tmr_stock)
	if cm_df['Actual'].iloc[i] == cm_df['Predicted'].iloc[i]:
		balance += difference * df["Close"].iloc[i + 1]
	else:
		balance -= difference * df["Close"].iloc[i + 1]
	i += 1
final_balance = round(balance, 2)
if balance > 125.7:
	print("Gaussian Naive Bayes ($" + str(final_balance) + ") was more effective than Buy-and-hold ($125.70) for year 2")
else:
	print("Gaussian Naive Bayes ($" + str(final_balance) + ") was less effective than Buy-and-hold ($125.70) for year 2")





