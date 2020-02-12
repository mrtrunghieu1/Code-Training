from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# Loading model 
MODEL = [LogisticRegression(random_state=0),
        GaussianNB(),
        KNeighborsClassifier(n_neighbors=10,p = 2)]
