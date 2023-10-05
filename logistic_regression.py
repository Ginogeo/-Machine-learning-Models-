
import numpy as np
class logistic_regression () :

    def __init__(self,learning_rate,no_of_iter ):
      self.learning_rate= learning_rate
      self.no_of_iter = no_of_iter

    def fit(self,x,y): 
      #no of data points in the data set (no of rows)=m
      #no of input features (no of columns)=n
      self.m, self.n=x.shape
      #initiatin weight and bias value
      self.w = np.zeros(self.n)
      self.b = 0
      self.x=x
      self.y=y
     # implimenting gradient descent
      for i in range (self.no_of_iter):
         self.update_weights ()  
            
    def update_weights(self) :
       #sigmoid funcn
       y_cap = 1/(1 + np.exp(-(self.x.dot(self.w)+self.b)))# h_cap = -1/(1 + e raised to -z) where z = wx+b
       #derivatives
       dw=(1/self.m)*np.dot(self.x.T, (y_cap - self.y))
       db = (1/self.m)*np.sum(y_cap - self.y)
       # updating w and b
       self.w = self.w-self.learning_rate*dw
       self.b = self.b - self.learning_rate*db

    def predict (self,x):
       y_pred = 1/(1 + np.exp(-(x.dot(self.w)+self.b)))
       #print(y_pred)
       y_pred = np.where(y_pred > 0.5 , 1, 0) 
       return y_pred
    
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
x_tr,x_ts,y_tr,y_ts = train_test_split(features,target,test_size=0.2,random_state=2)
#print(x_tr.shape)
#print(y_tr.shape)
classifier = logistic_regression(learning_rate=0.01,no_of_iter=1000)
classifier.fit(x_tr,y_tr)
x_tr_pred = classifier.predict(x_tr)
tr_data_accu = accuracy_score(y_tr,x_tr_pred)
print('accuracy of training data=', tr_data_accu)
x_ts_pred = classifier.predict(x_ts)
#print(x_ts.shape)
#print(y_ts.shape)
#print(x_ts_pred.shape)
ts_data_accu = accuracy_score(y_ts,x_ts_pred)
print("accuracy of test data=", ts_data_accu)
#predictive system
input_data = (5,116,74,0,0,25.6,0.201,30)

input_data_to_numpy = np.asarray(input_data)
reshaped_input_data = input_data_to_numpy.reshape(1,-1)
stda_data = scaler.transform(reshaped_input_data)
prediction = classifier.predict(stda_data)
print(prediction)
if( prediction [0] == 0):
   print('the person is not diabetic')
else:
   print('the person is diabetic')