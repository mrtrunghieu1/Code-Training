from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

# Loading model 
MODEL = [LogisticRegression(random_state=0),
        GaussianNB(),
        KNeighborsClassifier(n_neighbors=10,p = 2)]

#K-Fold
KF = KFold(n_splits=10)
