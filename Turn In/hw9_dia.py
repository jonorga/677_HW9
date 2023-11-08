###
### CS667 Data Science with Python, Homework 8, Jon Organ
###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

generate_all = True

# Question 1.1 ========================================================================================
print("Question 1.1:")
df = pd.read_csv("Diabetes dataset.csv")
df_0 = df[df['Outcome'] == 0]
df_1 = df[df['Outcome'] == 1]

print("File read into data frames...")


print("\n")
# Question 1.2 ========================================================================================
print("Question 1.2:")


def Q12(temp_df, outcome):
	df_corr = temp_df.corr().round(3)
	df_corr['X'] = df_corr.columns
	neworder = ['X', 'No', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
	       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
	df_corr = df_corr.reindex(columns=neworder)
	fig, ax = plt.subplots()
	fig.patch.set_visible(False)
	ax.axis('off')
	ax.axis('tight')
	ax.table(cellText=df_corr.values, colLabels=df_corr.columns, loc='center').scale(0.8,0.8)

	if generate_all:
		print("Saving Q1.2 " + outcome + " Table...\n")
		fig.savefig("results/Q1.2_" + outcome + "_Table.png", dpi = 600) #, dpi=1200
	else:
		print("Saving Q1.2 " + outcome + " Table skipped...\n")


Q12(df_0, "0")
Q12(df_1, "1")


print("\n")
# Question 1.3 ========================================================================================
print("Question 1.3:")
print("    a) Pregnancies and age have the highest correlation for healthy patients.")
print("    b) Skin thickness and age have the lowest correlation for healthy patients.")
print("    c) Insulin and skin thickness have the highest correlation for unhealthy patients.")
print("    d) BMI and pregnancies have the lowest correlation for unhealthy patients.")
print("    e) No, the correlated features are entirely different for each case.")


print("\n")
# Question 1.4 ========================================================================================
print("Question 1.4:")
print("Feature table:")
print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format('Class', 'μ(F1)', 'σ(F1)', 
	'μ(F2)', 'σ(F2)', 'μ(F3)', 'σ(F3)', 'μ(F4)', 'σ(F4)'))
print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format('0', 
	round(df_0['Glucose'].mean(), 2), round(df_0['Glucose'].std(), 2), 
	round(df_0['BloodPressure'].mean(), 2), round(df_0['BloodPressure'].std(), 2), 
	round(df_0['SkinThickness'].mean(), 2), round(df_0['SkinThickness'].std(), 2), 
	round(df_0['Insulin'].mean(), 2), round(df_0['Insulin'].std(), 2), ))
print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format('1', 
	round(df_1['Glucose'].mean(), 2), round(df_1['Glucose'].std(), 2), 
	round(df_1['BloodPressure'].mean(), 2), round(df_1['BloodPressure'].std(), 2), 
	round(df_1['SkinThickness'].mean(), 2), round(df_1['SkinThickness'].std(), 2), 
	round(df_1['Insulin'].mean(), 2), round(df_1['Insulin'].std(), 2), ))
print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format('All', 
	round(df['Glucose'].mean(), 2), round(df['Glucose'].std(), 2), 
	round(df['BloodPressure'].mean(), 2), round(df['BloodPressure'].std(), 2), 
	round(df['SkinThickness'].mean(), 2), round(df['SkinThickness'].std(), 2), 
	round(df['Insulin'].mean(), 2), round(df['Insulin'].std(), 2), ))

# F1 = Glucose
# F2 = BloodPressure
# F3 = SkinThickness
# F4 = Insulin

print("\n")
# Question 1.5 ========================================================================================
print("Question 1.5:")
print("The mean of each of the features show a clear pattern with the mean of the all class"
	" acting as a simple classifier for each feature. The deviations don't show as much of a pattern")

print("\n")
# Question 2 ========================================================================================
print("Question 2: Does not appear to be in this assignment")


print("\n")
# Question 3.1 ========================================================================================
print("Question 3.1:")
Y = df['Outcome']
X = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state = 0)


lr = LogisticRegression(solver='liblinear', random_state=0)
lr.fit(X_train, Y_train)

lr_predict = lr.predict(X_test)
lr_cm = pd.crosstab(Y_test, lr_predict)


print("Logistic regression classifier applied...")


print("\n")
# Question 3.2 ========================================================================================
print("Question 3.2:")
kNN1 = KNeighborsClassifier(n_neighbors = 1)
kNN3 = KNeighborsClassifier(n_neighbors = 3)
kNN5 = KNeighborsClassifier(n_neighbors = 5)

kNN1.fit(X_train, Y_train)
kNN3.fit(X_train, Y_train)
kNN5.fit(X_train, Y_train)

kNN1_predict = kNN1.predict(X_test)
kNN3_predict = kNN3.predict(X_test)
kNN5_predict = kNN5.predict(X_test)

kNN1_cm = pd.crosstab(Y_test, kNN1_predict)
kNN3_cm = pd.crosstab(Y_test, kNN3_predict)
kNN5_cm = pd.crosstab(Y_test, kNN5_predict)

print("kNN classifier applied...")


print("\n")
# Question 3.3 ========================================================================================
print("Question 3.3:")

nb = GaussianNB()
nb.fit(X_train, Y_train)

nb_predict = nb.predict(X_test)
nb_cm = pd.crosstab(Y_test, nb_predict)

print("Naive Bayesian classifier applied...")


print("\n")
# Question 3.4 ========================================================================================
print("Question 3.4:")
lda = LDA()
lda.fit(X_train, Y_train)

lda_predict = lda.predict(X_test)
lda_cm = pd.crosstab(Y_test, lda_predict)

print("Linear Discriminant Analysis classifier applied...")


print("\n")
# Question 3.5 ========================================================================================
print("Question 3.5:")
qda = QDA()
qda.fit(X_train, Y_train)

qda_predict = qda.predict(X_test)
qda_cm = pd.crosstab(Y_test, qda_predict)

print("Quadratic Discriminant Analysis classifier applied...")


print("\n")
# Question 3.6 ========================================================================================
print("Question 3.6:")
print("{:<16} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format('Method' ,'TP', 'FP', 'TN', 'FN', 
	'accuracy', 'TPR', 'TNR'))

def Q36(cm, model, method):
	TP = cm[0].iloc[0]
	FP = cm[0].iloc[1]
	TN = cm[1].iloc[1]
	FN = cm[1].iloc[0]
	acc = round(model.score(X_test, Y_test), 2)
	TPR = round(TP / (TP + FN), 2)
	TNR = round(TN / (TN + FP), 2)
	print("{:<16} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(method ,TP, FP, TN, FN, 
		acc, TPR, TNR))


Q36(lr_cm, lr, "Logistic Reg.")
Q36(kNN1_cm, kNN1, "k-NN (k = 1)")
Q36(kNN3_cm, kNN3, "k-NN (k = 3)")
Q36(kNN5_cm, kNN5, "k-NN (k = 5)")
Q36(nb_cm, nb, "Naive Bayesian")
Q36(lda_cm, lda, "Linear Discr.")
Q36(qda_cm, qda, "Quadr. Discr.")


print("\n")
# Question 3.7 ========================================================================================
print("Question 3.7:")
print("Surprisingly enough all of the different classifiers had reletively similar performances. The"
	" largest discrepancy in accuracy is 6%, they all have TPRs of somewhere between 80-90% and TNRs"
	" somewhere between 40-50%")


