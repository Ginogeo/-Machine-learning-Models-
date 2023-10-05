import numpy as np

class linear_Regression () :
   # initiating the parameters
   def __init__ (self, learning_rate, no_of_iter):
     self.learning_rate = learning_rate
     self. no_of_iter = no_of_iter

   def fit(self,x,y):
      # no of training examples
      self.m, self.n= x.shape #no of rows and columns # type: ignore
      # initiating the weight and bias of our model
      self.w=np.zeros(self.n)
      self.b=0
      self.x=x
      self.y=y
      #implementig grdient descent
      for i in range(self.no_of_iter):
         self.update_weights()
   def update_weights(self):  
      y_predict = self.predict(self.x) 
      #calaulate the gradient
      dw = -(2*(self.x.T).dot(self.y - y_predict))/self.m
      db = -2*np.sum(self.y-y_predict)/self.m
      # updating the weights
      self.w=self.w - self.learning_rate*dw
      self.b = self.b - self.learning_rate*db
   def predict(self,x):
      return x.dot(self.w)+self.b #x=wx+b
# using linear regression for prediction
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
sal_data = pd.read_csv('C:/Users/MY BOOK/Downloads/salary_data.csv') 
print(sal_data.head())
print(sal_data.shape)
print(sal_data.isnull().sum())
x= sal_data.iloc[:,:-1].values 
y= sal_data.iloc[:,1].values
x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.33,random_state=2)
model = linear_Regression(learning_rate=0.02,no_of_iter = 100)
model.fit(x_tr,y_tr)
print('weight= ' ,model.w[0])
print('bias=', model.b)
#y=9514*exp+23697
#predicting
x_ts_pred = model.predict(x_ts)
print(x_ts_pred)
(plt.scatter(x_ts,y_ts,color='red'))
(plt.plot(x_ts,x_ts_pred,color='blue'))
(plt.xlabel('workexp'))
(plt.title('sal vs exp'))
plt.show()
