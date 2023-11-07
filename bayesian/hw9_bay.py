###
### CS667 Data Science with Python, Homework 8, Jon Organ
###

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("cmg_weeks.csv")

X = df[['Avg_Return', 'Volatility']][df['Week'] <= 50].values
Y = df['Color'][df['Week'] <= 50].values

NB_classifier = GaussianNB().fit(X, Y)
prediction = NB_classifier.predict([[0.011720, 0.006433]])
print(prediction)

