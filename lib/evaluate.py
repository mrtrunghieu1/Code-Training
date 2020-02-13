import numpy as np
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
    def  