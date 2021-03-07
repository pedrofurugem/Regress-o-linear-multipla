import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from math import isnan
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing  import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score

dataset = pd.read_csv('weatherAUS.csv')

registers = dataset.iloc[:, [12, 14, 16, 18, 20, 22]].dropna() #excluiu os valores NaN
print(registers.shape)

X = registers.iloc[:, :-1]
Y = pd.get_dummies(registers['RainTomorrow']) #dividi em duas colunas

x = X.values.reshape(-1, 5)
y = Y.values.reshape(-1, 2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state=0)

lr = LinearRegression()
lr.fit(x_train, y_train)

prediction = lr.predict(x_test)

pred = np.argmax(prediction, axis = 1)
test = np.argmax(y_test, axis = 1)

print(precision_score(test, pred, average = 'micro'))

print(lr.coef_)