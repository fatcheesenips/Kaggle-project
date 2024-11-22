#first split then scale the data
import pandas as pd
import numpy as np
from matplotlib import pyplot 


train = pd.read_csv("train.csv")
train.isnull().sum()

train.dropna(subset=['Embarked'], inplace=True)
#train.dropna(subset=['Embarked'], inplace=True)

#use onehotencode on embarked

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(transform="pandas")
ohe_data = ohe.fit_transform(train[["Embarked"]])

train = train.drop(["Embarked"], axis=1)

train = pd.concat([train, ohe_data], axis=1)


# remove useless columns
train = train.drop(['PassengerId'], axis=1)
train = train.drop(['Name'], axis=1)
train = train.drop(["Ticket"], axis=1)
train = train.drop(["Cabin"], axis=1)

# use LabelEncoder for Sex
# Male / Female => 0/1
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
l1.fit(train["Sex"])
train["Sex"] = l1.transform(train["Sex"])


y = train["Survived"]

x = train.drop(['Survived'], axis=1)

from sklearn.impute import SimpleImputer
# this will replace the missing data with the average
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x.iloc[:, 2:4] = imputer.fit_transform(x.iloc[:, 2:4])

#fill missing data using simpleimputer
# split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=1)


#scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().set_output(transform="pandas")

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# apply logisitc regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")