###
### CS667 Data Science with Python, Homework 8, Jon Organ
###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#pd.set_option('display.max_colwidth', -1)
#pd.set_option('display.max_columns', None)


# Question 1.1 ========================================================================================
print("Question 1.1:")
df = pd.read_csv("Diabetes dataset.csv")
df_0 = df[df['Outcome'] == 0]
df_1 = df[df['Outcome'] == 1]

print("File read into data frames...")


print("\n")
# Question 1.2 ========================================================================================
print("Question 1.2:")



df_0_corr = df_0.corr().round(3)
df_0_corr['X'] = df_0_corr.columns
neworder = ['X', 'No', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome', 'X']
df_0_corr = df_0_corr.reindex(columns=neworder)
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax.table(cellText=df_0_corr.values, colLabels=df_0_corr.columns, loc='center').scale(1,1)

print("Saving Q1.2 Table...\n")
fig.savefig("Q1.2_Table.png", dpi = 600) #, dpi=1200