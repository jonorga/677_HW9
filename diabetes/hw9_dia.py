###
### CS667 Data Science with Python, Homework 8, Jon Organ
###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#pd.set_option('display.max_colwidth', -1)
#pd.set_option('display.max_columns', None)

generate_all = False

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



