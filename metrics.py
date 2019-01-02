import numpy as np

import pandas as pd
import copy
THRESHOLD = 15

def get_average_metrics(results):
    individual_performance = {}
    overall_performance = {}
    measure_list = ['recall', 'precision', 'F1',
                        'accuracy', 're', 'mae',
                        'maep', 'nde', 'sae']
    for appliance in results.keys():
        measure_dict = {}
        measure_average = {}
        # initialization
        for measure in measure_list:
            measure_dict[measure] = []
            overall_performance[measure] = []
            
        # save details
        for test_house in results[appliance]['y_test_raw'].keys():
            
            performance = get_all_metrics(results[appliance]['y_test_raw'][test_house],
                                              results[appliance]['pred_test'][test_house])
            for measure in performance.keys():
                measure_dict[measure].append(performance[measure])
        
        # save mean
        for measure in  measure_list:
            measure_average[measure] = np.mean(measure_dict[measure])
        individual_performance[appliance] = measure_average
        
    
    
    
    overall_performance_detail = {}

    # initialize
    for measure in  measure_list:
        overall_performance_detail[measure] = []
    
    # save details
    for appliance in individual_performance.keys():
        for measure in measure_list:
            overall_performance_detail[measure].append(individual_performance[appliance][measure])
    
    # save mean
    for measure in measure_list:
        overall_performance[measure] = np.mean(overall_performance_detail[measure])
    
    individual_performance = pd.DataFrame(individual_performance)
    return individual_performance, overall_performance
            
        


def get_all_metrics(target, prediction):
    threshold = THRESHOLD
    results = {'recall': get_recall(target, prediction, threshold),
               'precision': get_precision(target, prediction, threshold),
               'F1': get_F1(target, prediction, threshold),
               'accuracy': get_accuracy(target, prediction, threshold),
               're': get_relative_error(target, prediction),
               'mae': get_abs_error(target, prediction),
               'maep': get_abs_error_positive(target, prediction),
               'nde': get_nde(target, prediction),
               'sae': get_sae(target, prediction)}
    return results
    
def get_TP(target, prediction, threshold):
    '''
    compute the  number of true positive

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    assert (target.shape == prediction.shape)

    target = 1 - np.clip(target, threshold, 0) / threshold
    prediction = 1 - np.clip(prediction, threshold, 0) / threshold

    tp_array = np.logical_and(target, prediction) * 1.0
    tp = np.sum(tp_array)

    return tp


def get_FP(target, prediction, threshold):
    '''
    compute the  number of false positive

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    assert (target.shape == prediction.shape)

    target = np.clip(target, threshold, 0) / threshold
    prediction = 1 - np.clip(prediction, threshold, 0) / threshold

    fp_array = np.logical_and(target, prediction) * 1.0
    fp = np.sum(fp_array)

    return fp


def get_FN(target, prediction, threshold):
    '''
    compute the  number of false negtive

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    assert (target.shape == prediction.shape)

    target = 1 - np.clip(target, threshold, 0) / threshold
    prediction = np.clip(prediction, threshold, 0) / threshold

    fn_array = np.logical_and(target, prediction) * 1.0
    fn = np.sum(fn_array)

    return fn


def get_TN(target, prediction, threshold):
    '''
    compute the  number of true negative

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    assert (target.shape == prediction.shape)

    target = np.clip(target, threshold, 0) / threshold
    prediction = np.clip(prediction, threshold, 0) / threshold

    tn_array = np.logical_and(target, prediction) * 1.0
    tn = np.sum(tn_array)

    return tn


def get_recall(target, prediction, threshold):
    '''
    compute the recall rate

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    tp = get_TP(target, prediction, threshold)
    fn = get_FN(target, prediction, threshold)
#     print('tp={0}'.format(tp))
#     print('fn={0}'.format(fn))
    if tp + fn <= 0.0:
        recall = tp / (tp + fn + 1e-9)
    else:
        recall = tp / (tp + fn)
    return recall


def get_precision(target, prediction, threshold):
    '''
    compute the  precision rate

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    tp = get_TP(target, prediction, threshold)
    fp = get_FP(target, prediction, threshold)
#     print('tp={0}'.format(tp))
#     print('fp={0}'.format(fp))
    if tp + fp <= 0.0:
        precision = tp / (tp + fp + 1e-9)
    else:
        precision = tp / (tp + fp)
    return precision


def get_F1(target, prediction, threshold):
    '''
    compute the  F1 score

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    recall = get_recall(target, prediction, threshold)
#     print(recall)
    precision = get_precision(target, prediction, threshold)
#     print(precision)
    if precision == 0.0 or recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1


def get_accuracy(target, prediction, threshold):
    '''
    compute the accuracy rate

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    tp = get_TP(target, prediction, threshold)
    tn = get_TN(target, prediction, threshold)

    accuracy = (tp + tn) / target.size

    return accuracy


def get_relative_error(target, prediction):
    '''
    compute the  relative_error

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    '''

    assert (target.shape == prediction.shape)

    return np.mean(np.nan_to_num(np.abs(target - prediction) / np.maximum(target, prediction)))


def get_abs_error(target, prediction):
    '''
    compute the  absolute_error

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    '''

    assert (target.shape == prediction.shape)

    return np.mean(np.abs(target - prediction))

def get_abs_error_positive(target, prediction):
    '''
    compute the  absolute_error

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    '''

    
    assert (target.shape == prediction.shape)
    temp = copy.deepcopy(prediction)
    temp[temp < 0] = 0
    return np.mean(np.abs(target - temp))

def get_sum_ae(target, prediction):
    '''
    compute the  absolute_error

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    '''

    
    temp = copy.deepcopy(prediction)
    temp[temp < 0] = 0
    return np.sum(np.abs(target - temp))

def get_nde(target, prediction):
    '''
    compute the  normalized disaggregation error

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    '''

    return np.sum((target - prediction) ** 2) / np.sum((target ** 2))


def get_sae(target, prediction):
    '''
    compute the signal aggregate error
    sae = |\hat(r)-r|/r where r is the ground truth total energy;
    \hat(r) is the predicted total energy.
    '''
    r = np.sum(target)
    rhat = np.sum(prediction)

    return np.abs(r - rhat) / np.abs(r)