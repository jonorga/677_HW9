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

prediction_lda = LDA.predict(df[['Avg_Return', 'Volatility']][(df['Week'] > 50) & (df['Week'] <= 100)].values)
prediction_qda = QDA.predict(df[['Avg_Return', 'Volatility']][(df['Week'] > 50) & (df['Week'] <= 100)].values)
actual = df['Color'][(df['Week'] > 50) & (df['Week'] <= 100)].values

cm_df_lda = pd.DataFrame()
cm_df_lda['Actual'] = actual
cm_df_lda['Predicted'] = prediction_lda

cm_df_qda = pd.DataFrame()
cm_df_qda['Actual'] = actual
cm_df_qda['Predicted'] = prediction_qda

confusion_matrix_lda = pd.crosstab(cm_df_lda['Actual'], cm_df_lda['Predicted'])
confusion_matrix_qda = pd.crosstab(cm_df_qda['Actual'], cm_df_qda['Predicted'])

print("LDA confusion matrix:")
print(confusion_matrix_lda)
print("\nQDA confusion matrix:")
print(confusion_matrix_qda)


print("\n")
# Question 4 ========================================================================================
print("Question 4:")

def Q3(confusion_matrix, classifier):
	TP = confusion_matrix['Green'].iloc[0]
	TN = confusion_matrix['Red'].iloc[1]
	FP = confusion_matrix['Green'].iloc[1]
	FN = confusion_matrix['Red'].iloc[0]

	TPR = round((TP / (TP + FN) * 100), 2)
	TNR = round((TN / (TN + FP) * 100), 2)
	print(classifier, "classifier TPR and TNR")
	print("True positive rate: " + str(TPR) + "%")
	print("True negative rate: " + str(TNR) + "%\n")

Q3(confusion_matrix_lda, "LDA")
Q3(confusion_matrix_qda, "QDA")


print("\n")
# Question 5 ========================================================================================
print("Question 5:")
print("Buy-and-hold retrieved from previous assignment: $125.70")

def Q5(cmdf):
	balance = 100
	file_len = len(cmdf.index)
	i = 0
	while i < file_len:
		today_stock = balance / df['Close'].iloc[i]
		tmr_stock = balance / df['Close'].iloc[i + 1]
		difference = abs(today_stock - tmr_stock)
		if cmdf['Actual'].iloc[i] == cmdf['Predicted'].iloc[i]:
			balance += difference * df["Close"].iloc[i + 1]
		else:
			balance -= difference * df["Close"].iloc[i + 1]
		i += 1
	return round(balance, 2)

lda_balance = Q5(cm_df_lda)
qda_balance = Q5(cm_df_qda)
print("LDA strategy final balance: $" + str(lda_balance))
print("QDA strategy final balance: $" + str(qda_balance))
print("The QDA strategy was the most profitable")



