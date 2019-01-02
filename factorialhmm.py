from __future__ import print_function, division
import itertools
from copy import deepcopy
from collections import OrderedDict
from warnings import warn

import nilmtk
import pandas as pd
import numpy as np
import metrics
import matplotlib.pyplot as plt
from hmmlearn import hmm

from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator

# Python 2/3 compatibility
from six import iteritems
from builtins import range

SEED = 42

# Fix the seed for repeatibility of experiments
np.random.seed(SEED)

def sort_startprob(mapping, startprob):
    """ Sort the startprob according to power means; as returned by mapping
    """
    num_elements = len(startprob)
    new_startprob = np.zeros(num_elements)
    for i in range(len(startprob)):
        new_startprob[i] = startprob[mapping[i]]
    return new_startprob


def sort_covars(mapping, covars):
    new_covars = np.zeros_like(covars)
    for i in range(len(covars)):
        new_covars[i] = covars[mapping[i]]
    return new_covars


def sort_transition_matrix(mapping, A):
    """Sorts the transition matrix according to increasing order of
    power means; as returned by mapping

    Parameters
    ----------
    mapping :
    A : numpy.array of shape (k, k)
        transition matrix
    """
    num_elements = len(A)
    A_new = np.zeros((num_elements, num_elements))
    for i in range(num_elements):
        for j in range(num_elements):
            A_new[i, j] = A[mapping[i], mapping[j]]
    return A_new


def sort_learnt_parameters(startprob, means, covars, transmat):
    mapping = return_sorting_mapping(means)
    means_new = np.sort(means, axis=0)
    startprob_new = sort_startprob(mapping, startprob)
    covars_new = sort_covars(mapping, covars)
    transmat_new = sort_transition_matrix(mapping, transmat)
    assert np.shape(means_new) == np.shape(means)
    assert np.shape(startprob_new) == np.shape(startprob)
    assert np.shape(transmat_new) == np.shape(transmat)

    return [startprob_new, means_new, covars_new, transmat_new]


def compute_A_fhmm(list_A):
    """
    Parameters
    -----------
    list_pi : List of PI's of individual learnt HMMs

    Returns
    --------
    result : Combined Pi for the FHMM
    """
    result = list_A[0]
    for i in range(len(list_A) - 1):
        result = np.kron(result, list_A[i + 1])
    return result


def compute_means_fhmm(list_means):
    """
    Returns
    -------
    [mu, cov]
    """
    states_combination = list(itertools.product(*list_means))
    num_combinations = len(states_combination)
    means_stacked = np.array([sum(x) for x in states_combination])
    means = np.reshape(means_stacked, (num_combinations, 1))
    cov = np.tile(5 * np.identity(1), (num_combinations, 1, 1))
    return [means, cov]


def compute_pi_fhmm(list_pi):
    """
    Parameters
    -----------
    list_pi : List of PI's of individual learnt HMMs

    Returns
    -------
    result : Combined Pi for the FHMM
    """
    result = list_pi[0]
    for i in range(len(list_pi) - 1):
        result = np.kron(result, list_pi[i + 1])
    return result


def create_combined_hmm(model):
    list_pi = [model[appliance].startprob_ for appliance in model]
    list_A = [model[appliance].transmat_ for appliance in model]
    list_means = [model[appliance].means_.flatten().tolist()
                  for appliance in model]

    pi_combined = compute_pi_fhmm(list_pi)
    A_combined = compute_A_fhmm(list_A)
    [mean_combined, cov_combined] = compute_means_fhmm(list_means)

#     combined_model = hmm.GaussianHMM(
#         n_components=len(pi_combined), covariance_type='full',
#         startprob=pi_combined, transmat=A_combined)
    combined_model = hmm.GaussianHMM(n_components=len(pi_combined), covariance_type='full')
    combined_model.startprob_ = pi_combined
    combined_model.transmat_ = A_combined
    combined_model.covars_ = cov_combined
    combined_model.means_ = mean_combined
    
    return combined_model


def return_sorting_mapping(means):
    means_copy = deepcopy(means)
    means_copy = np.sort(means_copy, axis=0)

    # Finding mapping
    mapping = {}
    for i, val in enumerate(means_copy):
        mapping[i] = np.where(val == means)[0][0]
    return mapping


def decode_hmm(length_sequence, centroids, appliance_list, states):
    """
    Decodes the HMM state sequence
    """
    hmm_states = {}
    hmm_power = {}
    total_num_combinations = 1

    for appliance in appliance_list:
        total_num_combinations *= len(centroids[appliance])

    for appliance in appliance_list:
        hmm_states[appliance] = np.zeros(length_sequence, dtype=np.int)
        hmm_power[appliance] = np.zeros(length_sequence)

    for i in range(length_sequence):

        factor = total_num_combinations
        for appliance in appliance_list:
            # assuming integer division (will cause errors in Python 3x)
            factor = factor // len(centroids[appliance])

            temp = int(states[i]) / factor
            hmm_states[appliance][i] = temp % len(centroids[appliance])
            hmm_power[appliance][i] = centroids[
                appliance][hmm_states[appliance][i]]
    return [hmm_states, hmm_power]

class FHMM(Disaggregator):
    """
    Attributes
    ----------
    model : dict
    predictions : pd.DataFrame()
    meters : list
    MIN_CHUNK_LENGTH : int
    """

    def __init__(self):
        self.model = {}
        self.predictions = pd.DataFrame()
        self.MIN_CHUNK_LENGTH = 100
        self.MODEL_NAME = 'FHMM'                    


    def data_converter(self, data):
        main = {}
        apps = {}
        self.appliance_list = data.appliance_list
        main['train'] = []
        main['test'] = []
        apps['train'] = {}
        apps['test'] = {}
        
        
        # initialization
        for usage_type in ['train', 'test']:
            for appliance in data.appliance_list:
                apps[usage_type][appliance] = []
            
        for usage_type in ['train', 'test']:
            for key in data.key[usage_type]:
                print(key)
                main[usage_type].append(data.seq[key]['main'])
                for appliance in data.appliance_list:
                    if appliance in data.seq[key].keys():
                        apps[usage_type][appliance].append(data.seq[key][appliance])
       
    
        self.main = main
        self.apps = apps
                    
        
        

    def train(self, min_activation = 0):
        x_train = self.main['train']
        y_train = self.apps['train']
        models = {}
        for appliance in self.appliance_list:
            print("Training for", appliance)
            o = []
            for seq in y_train[appliance]:
                #seq = seq.reshape((seq.shape[0], 1))
                activation = (seq > 10).sum() * 1.0 / len(seq)
                if activation > min_activation:
                    #o.append(seq)
                    o = o + list(seq)

            if len(o) >= 1:
                o = np.array(o).reshape((-1,1))
                mod = hmm.GaussianHMM(2, "full")
                mod.fit(o)
                models[appliance] = mod
                print("Means for %s are" % appliance)
                print(mod.means_)
            else:
                print("Not enough samples for %s" % appliance)

        new_learnt_models = OrderedDict()
        for appliance, appliance_model in iteritems(models):
            startprob, means, covars, transmat = sort_learnt_parameters(
                appliance_model.startprob_, appliance_model.means_,
                appliance_model.covars_, appliance_model.transmat_)
            new_learnt_models[appliance] = hmm.GaussianHMM(
                startprob.size, "full", startprob, transmat)
            new_learnt_models[appliance].means_ = means
            new_learnt_models[appliance].covars_ = covars
            new_learnt_models[appliance].startprob_ = startprob
            new_learnt_models[appliance].transmat_ = transmat

        learnt_model_combined = create_combined_hmm(new_learnt_models)
        self.individual = new_learnt_models
        self.model = learnt_model_combined
        self.x_train = x_train
        self.y_train = y_train
        

    def test(self):
        """Disaggregate the test data according to the model learnt previously

        Performs 1D FHMM disaggregation.

        For now assuming there is no missing data at this stage.
        """
        x_test = self.main['test']
        y_test = self.apps['test']
        y_pred = {}
        y_pred_states = {}
        
        # Array of learnt states
        learnt_states_array = []
        for x_seq in x_test:
            learnt_states_array.append(self.model.predict(x_seq.reshape(-1,1)))
        
        # Model
        means = OrderedDict()
        for elec_meter, model in iteritems(self.individual):
            print(elec_meter)
            means[elec_meter] = (
                model.means_.round().astype(int).flatten().tolist())
            means[elec_meter].sort()
            y_pred[elec_meter] = []
            y_pred_states[elec_meter] = []
            
        
        for learnt_states in learnt_states_array:
            [decoded_states, decoded_power] = decode_hmm(
                len(learnt_states), means, means.keys(), learnt_states)
            for elec_meter, model in iteritems(self.individual):
                y_pred[elec_meter].append(decoded_power[elec_meter])
                y_pred_states[elec_meter].append(decoded_states[elec_meter])
                
        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_state = y_pred_states

    def get_mae(self):
        self.mae = {}
        for appliance in self.y_test.keys():
            mae_temp = []
            for i in range(len(self.y_test[appliance])):
                mae_temp.append(metrics.get_abs_error(self.y_test[appliance][i], self.y_pred[appliance][i]))
            self.mae[appliance] = np.mean(mae_temp)
      
        print(self.mae)
        return self.mae
    
    def get_sae(self):
        self.sae = {}
        for appliance in self.y_test.keys():
            sae_temp = []
            for i in range(len(self.y_test[appliance])):
                sae_temp.append(metrics.get_sae(self.y_test[appliance][i], self.y_pred[appliance][i]))
            self.sae[appliance] = np.mean(sae_temp)
        
        print(self.sae)
        
        return self.sae
    
    def get_all_metrics(self):
        individual_performance = {}
        overall_performance = {}
        measure_list = ['recall', 'precision', 'F1',
                            'accuracy', 're', 'mae',
                            'maep', 'nde', 'sae']

        for appliance in self.y_test.keys():
            measure_dict = {}
            measure_average = {}
            # initialization
            for measure in measure_list:
                measure_dict[measure] = []
                overall_performance[measure] = []

            # save details
            for test_house in range(len(self.y_test[appliance])):
                plt.plot(self.y_test[appliance][test_house], label = 'gt')
                plt.plot(self.y_pred[appliance][test_house], label = 'pred')
                plt.legend()
                plt.title('test house' + str(test_house) + '_' + appliance)
                plt.show()
                performance = metrics.get_all_metrics(self.y_test[appliance][test_house],
                                                  self.y_pred[appliance][test_house])
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


