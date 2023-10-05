import numpy as np

class Lasso_Regression () :
   # initiating the parameters
   def __init__ (self, learning_rate, no_of_iter, lambda_parameter):
     self.learning_rate = learning_rate
     self. no_of_iter = no_of_iter
     self.lambda_parameter = lambda_parameter
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
      for i in range (self.n):
         dw = np.zeros(self.n)
         if self.w [i]>0:

           dw[i] = (-(2*(self.x[:,1].T).dot(self.y - y_predict))+self.lambda_parameter)/self.m
         else:
           dw[i] = (-(2*(self.x[:,1].T).dot(self.y - y_predict))-self.lambda_parameter)/self.m  
      db = -2*np.sum(self.y-y_predict)/self.m
      # updating the weights
      self.w=self.w - self.learning_rate*dw
      self.b = self.b - self.learning_rate*db
   def predict(self,x):
      return x.dot(self.w)+self.b #x=wx+b
