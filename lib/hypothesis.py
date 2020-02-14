import statistics
import scipy

def mean_rules(cv_file,arrErrSum,arrErrProd,arrErrMin,arrErrMax):
    #Caculate mean and variance Total Sum Rule
    mean_sum = statistics.mean(arrErrSum)
    variance_sum = statistics.variance(arrErrSum)
    #Caculate mean and variance Total Product Rule
    mean_product = statistics.mean(arrErrProd)
    variance_product = statistics.variance(arrErrProd)
    #Caculate mean and variance Total Min Rule
    mean_min = statistics.mean(arrErrMin)
    variance_min = statistics.variance(arrErrMin)
    #Caculate mean and variance Total Max Rule
    mean_max = statistics.mean(arrErrMax)
    variance_max = statistics.variance(arrErrMax)
    dict_data = {'Dataset':cv_file,'mean_sum':mean_sum,'variance_sum':variance_sum,
                   'mean_product':mean_product,'variance_product':variance_product,
                  'mean_min':mean_min,'variance_min':variance_min,
                  'mean_max':mean_max,'variance_max':variance_max}
    return dict_data

def win_compare_error(array1, array2):
#     win_result = []
    win_result = 0
    for i in range(len(array1)):
#         win_result.append(1 if array1[i] < array2[i] else 0)
        if (array1[i] < array2[i]):
            win_result += 1
    return win_result

def equal_compare_error(array1, array2):
    equal_result = 0
    for i in range(len(array1)):
        if (array1[i] == array2[i]):
            equal_result += 1
    return equal_result

def loss_compare_error(array1, array2):
    loss_result = 0
    for i in range(len(array1)):
        if (array1[i] > array2[i]):
            loss_result += 1
    return loss_result

def wilcoxon(cv_file, data, number_cv):
    print('------------ Kiem dinh Wilcoxon ------------')
    print('Data: ',cv_file)
    for i in range(number_cv):
        print(scipy.stats.wilcoxon(data['arrErrSum'][i],data['arrErrProd'][i],zero_method='wilcox'))

def wilcoxon_test(cv_file, data, number_cv):
    print('------------ Kiem dinh Wilcoxon ------------')
    print('Data: ',cv_file)
    for i in range(number_cv):
        print("----SUM AND SVM----")
        print(scipy.stats.wilcoxon(data['arrErrSum'][i],data['arrErrSvm'][i],zero_method='wilcox'))
        print("----SUM AND DECISION TREE----")
        print(scipy.stats.wilcoxon(data['arrErrSum'][i],data['arrErrTree'][i],zero_method='wilcox'))
        print("----SVM AND DECISION TREE----")
        print(scipy.stats.wilcoxon(data['arrErrSvm'][i],data['arrErrTree'][i],zero_method='wilcox'))