from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
cv_score_lr = cross_val_score(LogisticRegression(max_iter=1000),X ,Y ,cv = 5)