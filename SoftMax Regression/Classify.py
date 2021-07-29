from softmaxReg import SoftmaxRegression
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

file = "beer.csv"

df = pd.read_csv(file,delimiter=',')
num_examples,num_features = df.shape


X = df[['calorific_value','nitrogen','turbidity','alcohol','sugars','bitterness','colour','degree_of_fermentation']]

Y= df['style']

X = X.to_numpy()
Y = Y.to_numpy()


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33333)


X_train_samples, X_train_features = X_train.shape
X_train = X_train[:,0:X_train_features]



clf1 = SoftmaxRegression(learn_rate=0.0001,num_iterations=3000)
train = clf1.learn(X_train,Y_train)
preds = clf1.return_preds(X_test)
accuracy_score =clf1.accuracy(X_train,Y_train)


print("\nPredicted Values: \n",preds)
print("\nActual Values: \n",Y_test)
print("\nPercentage Correctly Classified: ","{:.2%}".format(accuracy_score))


pd.DataFrame(preds).to_csv("output.csv")

#pd.DataFrame(cla)