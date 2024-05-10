# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Split the data into training and testing sets.
5. convert the text data into a numerical representation using CountVectorizer.
6. Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
7. Finally, evaluate the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Aprajith R
RegisterNumber: 212222080006

import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy

*/
```

## Output:
![Screenshot 2024-05-10 215713](https://github.com/Aprajith-R/Implementation-of-SVM-For-Spam-Mail-Detection/assets/161153978/311d2aa3-9467-4d38-9f1a-2fd6e335e3f9)
![Screenshot 2024-05-10 215724](https://github.com/Aprajith-R/Implementation-of-SVM-For-Spam-Mail-Detection/assets/161153978/d1daa788-31bd-4280-8171-ab2d41e8720a)
![Screenshot 2024-05-10 215737](https://github.com/Aprajith-R/Implementation-of-SVM-For-Spam-Mail-Detection/assets/161153978/0470e127-d5d6-4777-8beb-21bafb426f1f)
![Screenshot 2024-05-10 215748](https://github.com/Aprajith-R/Implementation-of-SVM-For-Spam-Mail-Detection/assets/161153978/1680753e-bd30-4597-aeab-c9c8e6cffd6a)
![Screenshot 2024-05-10 215800](https://github.com/Aprajith-R/Implementation-of-SVM-For-Spam-Mail-Detection/assets/161153978/55aa37d6-e9d4-4139-aa67-e52dce841c73)
![Screenshot 2024-05-10 215809](https://github.com/Aprajith-R/Implementation-of-SVM-For-Spam-Mail-Detection/assets/161153978/f3eda291-e301-4580-b398-3987176b3f92)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
