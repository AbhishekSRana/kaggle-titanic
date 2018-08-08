#including the libraries 
import numpy as np
import pandas as pd
from sklearn.preprocessing import  Imputer, LabelEncoder ,OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

#importing the training data
data_train=pd.read_csv('train.csv')

#avoiding the inefficitve columns
df = pd.DataFrame(data_train, columns = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',  'Fare'])

missing_age = df[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch',  'Fare']]
  
  # Split into sets with given and notgiven Age values
Age_given    = missing_age.loc[ (df.Age.notnull()) ]
Age_notgiven = missing_age.loc[ (df.Age.isnull()) ]
   
    #  feature array of Age_given set
X = Age_given.iloc[:, 1:].values

 # target array of Age_given set
Y = Age_given.iloc[:, 0].values

#encodeing the categorical data of GENDER coloum
# female-> 0 and male -> 1 
labelencoder_X=LabelEncoder()
X[:,1]=labelencoder_X.fit_transform(X[:,1])
onehotencoder= OneHotEncoder(categorical_features=[1])
X= onehotencoder.fit_transform(X).toarray()
 
#avoiding the dummy variable column
X =X[:,1:]

#feature scaling of the column 3 and 4
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X[:,[3,4]]= sc_X.fit_transform(X[:,[3,4]])

# linear Regression Model for predicting the missing age

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X,Y)

# feature array of the Age_notgiven set
x = Age_notgiven.iloc[:, 1:].values

#encodeing the categorical data of GENDER coloum
# female-> 0 and male -> 1 
labelencoder_x=LabelEncoder()
x[:,1]=labelencoder_x.fit_transform(x[:,1])
onehotencoder_x= OneHotEncoder(categorical_features=[1])
x= onehotencoder_x.fit_transform(x).toarray()

#avoiding the dummy variable column 
x =x[:,1:]

#feature scaling of the column 3 and 4
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x[:,[3,4]]= sc_x.fit_transform(x[:,[3,4]])

# predicting the missing age value
predicted_missingAges = regressor.predict(x)
   
 # Assign those predictions to the full data set
df.loc[ (df.Age.isnull()), 'Age'] = predicted_missingAges


X_train= df.iloc[: ,2 : ].values
Y_train= df.iloc[:,1].values 



#encodeing the categorical data of GENDER coloum
# female-> 0 and male -> 1 
labelencoder_X=LabelEncoder()
X_train[:,1]=labelencoder_X.fit_transform(X_train[:,1])
onehotencoder= OneHotEncoder(categorical_features=[1])
X_train= onehotencoder.fit_transform(X_train).toarray()

#removing the dummy variable column
X_train=X_train[:,1:]






#importing the test data
data_test=pd.read_csv('test.csv')

df_1 = pd.DataFrame(data_test, columns = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',  'Fare'])

missing_age_test = df_1[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch',  'Fare']]
  
  # Split into sets with given and notgiven Age values
Age_given_test    = missing_age_test.loc[ (df_1.Age.notnull()) ]
Age_notgiven_test = missing_age_test.loc[ (df_1.Age.isnull()) ]
   
    #  feature array of Age_given set
X = Age_given_test.iloc[:, 1:].values

 # target array of Age_given set
Y = Age_given_test.iloc[:, 0].values


#filling the missing the data of Fare column
imputer=Imputer(missing_values="NaN", strategy="median", axis=0)
imputer=imputer.fit(X[:,4:5])
X[:,4:5]=imputer.transform(X[:,4:5])

#encodeing the categorical data of GENDER coloum
# female-> 0 and male -> 1 
labelencoder=LabelEncoder()
X[:,1]=labelencoder.fit_transform(X[:,1])
onehotencoder_x= OneHotEncoder(categorical_features=[1])
X= onehotencoder_x.fit_transform(X).toarray()
 
#avoiding the dummy variable column
X =X[:,1:]

#feature scaling of the column 3 and 4
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X[:,[3,4]]= sc_X.fit_transform(X[:,[3,4]])

# linear Regression Model for predicting the missing age

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X,Y)

# feature array of the Age_notgiven set
x = Age_notgiven_test.iloc[:, 1:].values

#encodeing the categorical data of GENDER coloum
# female-> 0 and male -> 1 
labelencoder_x=LabelEncoder()
x[:,1]=labelencoder_x.fit_transform(x[:,1])
onehotencoder_x= OneHotEncoder(categorical_features=[1])
x= onehotencoder_x.fit_transform(x).toarray()

#avoiding the dummy variable column 
x =x[:,1:]

#feature scaling of the column 3 and 4
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x[:,[3,4]]= sc_x.fit_transform(x[:,[3,4]])

# predicting the missing age value
predicted_missingAges_test = regressor.predict(x)
   
 # Assign those predictions to the full data set
df_1.loc[ (df_1.Age.isnull()), 'Age'] = predicted_missingAges_test


X_test= df_1.iloc[: ,1:].values
Y_test= df_1.iloc[: ,0 ].values


#filling the missing the data of Fare column
imputer_test=Imputer(missing_values="NaN", strategy="median", axis=0)
imputer_test=imputer_test.fit(X_test[:,5:6])
X_test[:,5:6]=imputer_test.transform(X_test[:,5:6])


# encoding the categorical data of GENDER coloum
# female-> 0 and male -> 1 
labelencoder_x_test=LabelEncoder()
X_test[:,1]=labelencoder_x_test.fit_transform(X_test[:,1])
onehotencoder_test= OneHotEncoder(categorical_features=[1])
X_test= onehotencoder_test.fit_transform(X_test).toarray()
 
#avoiding the dummy variable column
X_test=X_test[:,1:]


# decision tree model
classifier=DecisionTreeClassifier(criterion='gini')
classifier.fit(X_train,Y_train)

# predicted values of X_test
Y_pred=classifier.predict(X_test)

# prediction of the X_test 
prediction= np.column_stack((Y_test, Y_pred))

