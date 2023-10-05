import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

class SVM_classifier():
    #initiating hyperparameters
    def __init__(self, learning_rate, no_of_iter, lambda_parameter):
        self.learning_rate = learning_rate
        self.no_of_iter = no_of_iter
        self.lambda_parameter = lambda_parameter
        
    def fit(self,x,y):
        #x= input features , y = target
        self.m, self.n = x.shape
        #m - no of data points-no of rows()
        #n - no of input features - no of  columns(weight)
        self.w=np.zeros(self.n)#creating a array of zeros with shape of n
        self.b=0
        self.x=x
        self.y=y
        #implementing gradient descent
        for i in range (self.no_of_iter):
            self.update_weights()

    #updates weight and bias values
    def update_weights(self,):
        #label encoding
        y_label = np.where(self.y<=0,-1,1)#if label value is less than or equal to zero it converts all the values to -1 else it converts to 1
        for index, x_i in enumerate(self.x):
            # y value at [index] * ((x at [index] * weight) - bias )
            condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1
            if (condition == True):
                dw = 2* self.lambda_parameter*self.w
                db = 0
            else:
                dw=2*self.lambda_parameter*self.w - np.dot(x_i , y_label[index])
                db = y_label[index]
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db           
    def predict(self,x):
        output = np.dot(x,self.w ) - self.b
        predicted_labels = np.sign(output)#gives the predicted_labels as 1 or -1
        y_cap = np.where(predicted_labels <= -1,0,1) # all -1 values are changed to zero and 1 is kept as it is
        return y_cap
    


 