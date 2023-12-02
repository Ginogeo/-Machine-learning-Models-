import numpy as np
import pandas as pd
from sklearn.svm import SVC
import sklearn.datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
data = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data['label']= breast_cancer_dataset.target
#print(data.head())
#print(data.isnull().sum())
#print(data['label'].value_counts())
x = data.drop(columns='label',axis=1)
y = data['label']
x = np.asarray(x)
y = np.asarray(y)
# grid search is used to determine the best parameter for our model
model = SVC()
parameters  = {
    'kernel':['linear', 'poly','rbf','sigmoid'],
    'C':[1, 5, 10, 20]
}

classifier = GridSearchCV(model, parameters,cv=5)
# fitting data
classifier.fit(x,y)
print(classifier.cv_results_)
