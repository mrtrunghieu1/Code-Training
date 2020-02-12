from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors, datasets
import os 
import pickle
from constant import path 
class TrainMaster():
    def training_phase(self,model,features_train,targets_train):
        learner = model
        learner.fit(features_train,targets_train)
        model_path = os.path.join(path.MODEL_PATH,type(learner).__name__)
        pickle_out = open('{}'.format(model_path),"wb")
        pickle.dump(learner, pickle_out)
        pickle_out.close()
        print('----- Dump model {} to success! -----'.format(type(learner).__name__))
 
    def testing_phase(self,load_model,features_test):
        pickle_in = open(load_model,"rb")
        clf = pickle.load(pickle_in)
        print('load model success')
        predict_probability = clf.predict_proba(features_test)
        return predict_probability

        
