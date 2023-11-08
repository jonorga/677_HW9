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





