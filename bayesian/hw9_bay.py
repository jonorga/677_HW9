###
### CS667 Data Science with Python, Homework 8, Jon Organ
###

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("cmg_weeks.csv")

input_data = df[['Avg_Return', 'Volatility']][df['Week'] <= 50]
dummies = [pd.get_dummies(df[c]) for c in input_data.columns]
binary_data = pd.concat(dummies, axis=1)
X = binary_data[0:50].values
le = LabelEncoder()
Y = le.fit_transform(df['Color'][df['Week'] <= 50].values)
NB_classifier = MultinomialNB().fit(X, Y)


