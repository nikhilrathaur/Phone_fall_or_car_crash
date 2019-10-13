import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split


phone_fall=pd.read_excel('PHONE FALL DATA FILE.xls')
accidental_fall=pd.read_excel('ACCIDENTAL DATA FILE.xlsx')

#Giving labels to categorize the data
Zero= np.array(phone_fall.shape[0]*[0])
One= np.array(accidental_fall.shape[0]*[1])

phone_fall['Label']=Zero
accidental_fall.columns= ['abc', 'Accel X', 'Accel Y']
accidental_fall['Label']=One

#Extracting the necessary coulumns and appending thm together into one dataframe
Array1=phone_fall.iloc[:, 2:5]
Array2=accidental_fall.iloc[:, 1:4]
dataframe= pd.concat([Array1, Array2], ignore_index=True)

def training_model(x,y):
    ''' Training the model, find accuracy on testing data, predict output'''
    X = dataframe.iloc[:, 0:2]
    Y = dataframe.iloc[:, 2]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    model = svm.SVC()
    model.fit(x_train, y_train)
    result1 = model.score(x_test, y_test)
    print('Accuracy of model is: %f %%' % (result1 * 100))
    result2=model.predict([[x,y]])
    print("PREDICTION :")
    if result2==0:
        print("Phone has fallen")
    else:
        print("It is a car accident")
    return result2


if __name__ == '__main__':
    while True:
        x= float(input("Enter the x-acceleration"))
        y= float(input("Enter the y-acceleration"))
        result = training_model(x, y)
        result=float(result[0])
        a=[x,y,result]
        b1=a[0]
        b2=a[1]
        b3=a[2]
        a=pd.DataFrame(a)
        a=pd.DataFrame(a.T, columns = ['Accel X', 'Accel Y', 'Label'])
        a['Accel X']=b1
        a['Accel Y']=b2
        a['Label']=b3
        print(a)
        dataframe = pd.concat([dataframe, a], ignore_index=True)
        print(dataframe)
        print("Here you can see that the input given for testing is also appened to the dataframe and is used for training the next testing inputs")
        z=int(input("Enter 0 for exit or any other key to continue giving the inputs for testing..."))
        if z==0:
            exit(0)
