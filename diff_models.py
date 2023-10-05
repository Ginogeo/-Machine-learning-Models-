from sklearn.linear_model import LogisticRegression
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

diabetes_data = pd.read_csv('C:/Users/MY BOOK/Downloads/diabetes.csv')
#print(diabetes_data.head())
diabetes_data.groupby('Outcome').mean()
features = diabetes_data.drop(columns = 'Outcome', axis= 1)
target = diabetes_data['Outcome']
#print(target.shape)
scaler = StandardScaler()
scaler.fit(features)

std_data = scaler.transform(features)
#print(std_data)
features = std_data
target=diabetes_data['Outcome']
from sklearn.model_selection import cross_val_score
cv_score_lr = cross_val_score(LogisticRegression(max_iter=1000),features ,target ,cv = 5)
#print(cv_score_lr)
mean_acc = sum (cv_score_lr/len(cv_score_lr))
mean_acc= mean_acc*100
mean_acc= round(mean_acc,2)# rounding to rwo decimal places
print('k cross vlaidation -', mean_acc)
x_tr,x_ts,y_tr,y_ts = train_test_split(features,target,test_size=0.2,random_state=2)
models = [LogisticRegression(), SVC(kernel='linear'),KNeighborsClassifier(),RandomForestClassifier()]
def compare_models():
   for model in models:
      model.fit(x_tr,y_tr)
      x_tr_pred = model.predict(x_ts)
      tr_data_accu = accuracy_score(y_ts,x_tr_pred)
      print('accuracy of training data=',model,'-', tr_data_accu)
     
compare_models()

#print(y_tr.shape)

#x_tr_pred = classifier.predict(x_tr)
#tr_data_accu = accuracy_score(y_tr,x_tr_pred)
#print('accuracy of training data=', tr_data_accu)
#x_ts_pred = classifier.predict(x_ts)
#print(x_ts.shape)
#print(y_ts.shape)
#print(x_ts_pred.shape)
#ts_data_accu = accuracy_score(y_ts,x_ts_pred)
#print("accuracy of test data=", ts_data_accu)
#predictive system
#input_data = (5,116,74,0,0,25.6,0.201,30)

#input_data_to_numpy = np.asarray(input_data)
#reshaped_input_data = input_data_to_numpy.reshape(1,-1)
#stda_data = scaler.transform(reshaped_input_data)
#prediction = classifier.predict(stda_data)
#print(prediction)
#if( prediction [0] == 0):
   #print('the person is not diabetic')
#else:
   #print('the person is diabetic')