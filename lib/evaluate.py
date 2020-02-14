import numpy as np
import pandas as pd
import statistics
import pickle
import os
from constant import path, parameter

class Rules():
    #--------------------------------- Combining Sum Rule ---------------------------------
    def combining_sum_rule(self,meta_proba, len_models):
        matrix = np.hsplit(meta_proba, len_models)
        num_classes = int(meta_proba.shape[1] / len_models)
        combining_sum_rule = np.zeros((meta_proba.shape[0], num_classes))
        for i in range(len_models):
            combining_sum_rule += matrix[i] 
        combining_sum_rule *= 1/3
        return combining_sum_rule
    
    # --------------------------------- Combining Product Rule ---------------------------------
    def combining_product_rule(self,meta_proba, len_models):
        matrix = np.hsplit(meta_proba, len_models)
        num_classes = int(meta_proba.shape[1] / len_models)
        combining_product_rule = np.ones((meta_proba.shape[0], num_classes))
        for i in range(len_models):
            combining_product_rule *= matrix[i]
        return combining_product_rule
    
    # --------------------------------- Combining Max Min Rule ---------------------------------
    def combining_max_min(self, meta_proba, len_models, parameter):
        values = []
        matrix = np.hsplit(meta_proba, len_models)
        num_classes = int(meta_proba.shape[1] / len_models)
        combining_max_min = np.zeros((meta_proba.shape[0], num_classes))
        for i in range(combining_max_min.shape[0]):
            array = []
            for sub_matrix in matrix:
                array.append(sub_matrix[i])
                nparray = np.asarray(array)
            if parameter == 'max':
                values = nparray.max(0)
            else:
                values = nparray.min(0)
            combining_max_min[i] = values
        return combining_max_min
    
    # ------------------------ Determine target from combining rule ------------------------
    def target(self,combining_matrix, classes):
        targets_combining_algorithm = []
        df = pd.DataFrame(combining_matrix, columns = classes)
        targets = df.idxmax(axis = 1)
        return targets

    # ------------------------------ Error Combining Rule ----------------------------------
    def error_combining_rule(self,target_combining_rule, target_test):
        boolen_result = []
        for i in range(len(target_combining_rule)):
            result = 1 if target_combining_rule[i] != target_test[i] else 0
            boolen_result.append(result)
        mean_combining_rule = statistics.mean(boolen_result)
        return mean_combining_rule

    # ------------------------------ Save Results ------------------------------------------
    def writer_output(self,cv_file, arrErrSum, arrErrProd, arrErrMin, arrErrMax):
        pickle_file = {'Dataset':cv_file,'arrErrSum':arrErrSum,'arrErrProd':arrErrProd,
                   'arrErrMin':arrErrMin,'arrErrMax':arrErrMax}
        file_result = os.path.join(path.RESULT_PATH,"result_fix_{}.pickle".format(cv_file))
        pickle_out = open(file_result, "wb")
        pickle.dump(pickle_file, pickle_out)
        pickle_out.close()
        
