#first split then scale the data
import pandas as pd
import numpy as np
from matplotlib import pyplot 


train = pd.read_csv('train.csv')
print(train.head())

x = train.drop(['PassengerId'],['Survived'],['Pclass'], axis=1)
x = train.drop(['Survived'])
x = train.drop('Pclass')

y = train['Survived']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state= 1)

