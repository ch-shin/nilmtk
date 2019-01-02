import numpy as np
import pandas as pd
import os
import pickle
import random
from random import randint

DATA_DIR = '/home/chshin/kdd-19/data/'
params_appliance = {'kettle':{'windowlength':599,
                              'on_power_threshold':2000,
                              'max_on_power':3998,
                             'mean':700,
                             'std':1000,
                             's2s_length':128},
                    'microwave':{'windowlength':599,
                              'on_power_threshold':200,
                              'max_on_power':3969,
                                'mean':500,
                                'std':800,
                                's2s_length':128},
                    'fridge':{'windowlength':599,
                              'on_power_threshold':50,
                              'max_on_power':3323,
                             'mean':200,
                             'std':400,
                             's2s_length':512},
                    'dish_washer':{'windowlength':599,
                              'on_power_threshold':10,
                              'max_on_power':3964,
                                  'mean':700,
                                  'std':1000,
                                  's2s_length':1536},
                    'washing_machine':{'windowlength':599,
                              'on_power_threshold':20,
                              'max_on_power':3999,
                                      'mean':400,
                                      'std':700,
                                      's2s_length':2000}}
    

# def read_data(path):
#    return(np.genfromtxt(path, delimiter = ','))

def save_pickle(target, file_name):
    file_object = open(file_name,'wb')
    pickle.dump(target, file_object)
    file_object.close()
    
def load_pickle(file_name):
    file_object = open(file_name,'r')  
    loaded = pickle.load(file_object)
    return(loaded)

class DataHandler(object):
    def __init__(self, output_length = 32, additional_window = 200):
        self.output_length = output_length
        self.additional_window = additional_window
        self.window = output_length + (2 * additional_window)
        self.params_appliance = params_appliance
        self.normalized = False
        
    def read_data(self, data_name):
        if data_name.lower() == 'ukdale':
            self._read_ukdale()
        elif data_name.lower() == 'redd':
            self._read_redd()
        elif data_name.lower() == 'encored':
            self._read_encored()
        else:
            raise Exception("No data: " + data_name + "data_name should be in ['ukdale', 'redd', 'encored']")

    def _read_ukdale(self):
        self.data_name = 'ukdale'
        UKDALE_APPLIANCE_LIST = ['kettle', 'microwave', 'fridge', 'dish_washer', 'washing_machine']
        self.appliance_list = UKDALE_APPLIANCE_LIST
        self._HOUSE_LIST = range(1,6)
        self.key = {}
        self.key['train'] = [1, 3, 4, 5]
        self.key['test'] = [2]
        self.key_apps = {}
        self.key_apps['train'] = {}
        self.key_apps['test'] = {}
        UKDALE_PATH = DATA_DIR + 'ukdale/'
        
        file_list = os.listdir(UKDALE_PATH)
        num_appliances = len(self.appliance_list)
        num_house = len(self._HOUSE_LIST)
        
        seq = {}
        
        # load data into memory
        for i in self._HOUSE_LIST:
            seq[i] = {}
            for appliance in self.appliance_list:
                file_name = 'building_' + str(i) + '_' + appliance + '.csv'
                if file_name in file_list:
                    temp = pd.read_csv(UKDALE_PATH + file_name, header = None)
                    seq[i][appliance] = temp[0]
                    print(file_name, temp[0].shape)
                    if i in self.key['train']:
                        if appliance not in self.key_apps['train'].keys():
                            self.key_apps['train'][appliance] = []
                        self.key_apps['train'][appliance].append(i)
                    
                    if i in self.key['test']:
                        if appliance not in self.key_apps['test'].keys():
                            self.key_apps['test'][appliance] = []
                        self.key_apps['test'][appliance].append(i)
                        
            file_name = UKDALE_PATH + 'building_' + str(i) + '_mains.csv'
            temp = pd.read_csv(file_name, header = None)
            seq[i]['main'] = temp[0]
            print(file_name, temp[0].shape)
            
        # length fitting            
        for i in self._HOUSE_LIST:
            min_length = len(seq[i]['main'])
            for appliance in list(seq[i].keys()):
                min_length = min(len(seq[i][appliance]), min_length)
            
            for appliance in list(seq[i].keys()):
                seq[i][appliance] = seq[i][appliance][:min_length]
        
        self.seq = seq
        
    def _read_encored(self):
        self.data_name = 'encored'
        ENCORED_APPLIANCE_LIST = ['tv', 'fridge', 'air_conditioner', 'rice_cooker', 'washing_machine', 'microwave']
        self.appliance_list = ENCORED_APPLIANCE_LIST
        ENCORED_PATH = DATA_DIR + 'encored/'
        self._HOUSE_LIST = os.listdir(ENCORED_PATH + 'train/') + os.listdir(ENCORED_PATH + 'test/')
        num_appliances = len(self.appliance_list)
        self.key = {}
        self.key['train'] = []
        self.key['test'] = []
        self.key_apps = {}
        self.key_apps['train'] = {}
        self.key_apps['test'] = {}
        
        
        seq = {}
        for data_usage_type in os.listdir(ENCORED_PATH):
            for site in os.listdir(ENCORED_PATH + data_usage_type + '/'):
                for date in os.listdir(ENCORED_PATH + data_usage_type + '/' + site + '/'):
                    data_key = site + '_' + date
                    seq[data_key] = {}
                    self.key[data_usage_type].append(data_key)
                    main = pd.read_csv(ENCORED_PATH + data_usage_type + '/' + site + '/' + date + '/mains.csv', index_col = 0)
                    seq[data_key]['main'] = main['active_power']
                    print(data_key, 'main', seq[data_key]['main'].shape)
                    for appliance in self.appliance_list:
                        if (appliance + '.csv') in os.listdir(ENCORED_PATH + data_usage_type + '/' + site + '/' + date + '/'):
                            app_data = pd.read_csv(ENCORED_PATH + data_usage_type + '/' + site + '/' + date + '/' + appliance + '.csv', index_col = 0)
                            seq[data_key][appliance] = app_data['active_power']
                            print(data_key, appliance, seq[data_key][appliance].shape)
                            if appliance not in self.key_apps[data_usage_type].keys():
                                self.key_apps[data_usage_type][appliance] = []
                            self.key_apps[data_usage_type][appliance].append(data_key)
        self.seq = seq
        
    def _read_redd(self):
        self.data_name = 'redd'
        REDD_APPLIANCE_LIST = ['fridge', 'dish_washer', 'washing_machine', 'microwave']
        self.appliance_list = REDD_APPLIANCE_LIST
        num_appliances = len(self.appliance_list)
        self.key = {}
        self.key['train'] = []
        self.key['test'] = []
        self.key_apps = {}
        self.key_apps['train'] = {}
        self.key_apps['test'] = {}
        seq = {}
        
        houses = {'train': ['house2', 'house3', 'house4', 'house5', 'house6'],
                  'test': ['house1']}

        for appliance in self.appliance_list:
            self.key_apps['train'][appliance] = []
            self.key_apps['test'][appliance] = []
        
        REDD_PATH = DATA_DIR + 'redd/'
        for file_name  in os.listdir(REDD_PATH):
            
            file_path = REDD_PATH + file_name
            data = pd.read_csv(file_path, index_col = 0)
            data.columns = [col_name.replace(' ', '_') for col_name in data.columns]
            data.columns = [col_name.replace('washer_dryer', 'washing_machine') for col_name in data.columns]
            key = file_name.split('.')[0]
            seq[key] = {}
            seq[key]['main'] = data['main']
            
            # nan handling
            seq[key]['main'] = seq[key]['main'].fillna(method = 'bfill')
            for usage_type in ['train', 'test']:
                house_list = houses[usage_type]
                for house in house_list:
                    if house in key:
                        self.key[usage_type].append(key)
            
            for appliance in self.appliance_list:
                if appliance in data.columns:
                    seq[key][appliance] = data[appliance]
                    # nan handling
                    seq[key][appliance] = seq[key][appliance].fillna(method = 'bfill')
                    for usage_type in ['train', 'test']:
                        house_list = houses[usage_type]
                        for house in house_list:
                            if house in key:
                                self.key_apps[usage_type][appliance].append(key)
        self.seq = seq
                            
            
    def normalize(self, normalize_type = 'ms', debug = False):
        self.normalize_type = normalize_type
        
        power_sum = 0
        len_sum = 0
        for key_train in self.key['train']:
            power_sum += np.sum(self.seq[key_train]['main'])
            len_sum += len(self.seq[key_train]['main'])
            # print('power_sum', power_sum, 'len_sum', len_sum)
        
        self.mean = power_sum / len_sum
        square_sum = 0
        
        for key_train in self.key['train']:
            square_sum += np.sum(np.power(self.seq[key_train]['main'] - self.mean, 2))
            # print('square_sum', square_sum)
            
        self.std = np.sqrt(square_sum / len_sum)
        print('mean', self.mean, 'std', 'std', self.std)
        for data_usage_type in ['train', 'test']:
            for key in self.key[data_usage_type]:
                for app in self.seq[key].keys():
                    if normalize_type == 'ms':
                        self.seq[key][app] = (self.seq[key][app] - self.mean) / self.std
                    else:
                        self.seq[key][app] = (self.seq[key][app]) / self.std
        
        self.normalized = True
            
            
class BatchGenerator(object):
    def __init__(self, data, target_appliance, batch_size = 16):
        self._data = data
        self.target_appliance = target_appliance
        self.batch_size = batch_size
        self.output_length = data.output_length
        self.additional_window = data.additional_window
        self.end = False
                
    def next_batch(self):
        # random batch generation
        # 1) randomly select a key(house & date)
        # 2) randomly select window
        input_length = self.output_length + (2 * self.additional_window)
        x_batch = np.zeros((self.batch_size, input_length, 1))
        y_batch = np.zeros((self.batch_size, self.output_length, 1))
        random_key = random.choice(self._data.key_apps['train'][self.target_appliance])
        x_seq = self._data.seq[random_key]['main']
        y_seq = self._data.seq[random_key][self.target_appliance]
        
        # TODO: code optimization
        random_index_list = np.random.randint(0, len(x_seq) - self.output_length, self.batch_size)
        for i, random_index in enumerate(random_index_list):
            y_batch[i] = np.reshape(y_seq[random_index:(random_index + self.output_length)], (-1, 1))
            if random_index + self.output_length + self.additional_window > len(x_seq):
                filled_length = len(x_seq) - (random_index - self.additional_window)
                x_batch[i,:filled_length] = np.reshape(x_seq[(random_index - self.additional_window):len(x_seq)], (-1, 1))
            elif random_index - self.additional_window < 0:
                filled_length = random_index + self.output_length + self.additional_window
                backward_filled_length = input_length - filled_length
                x_batch[i, backward_filled_length:]= np.reshape(x_seq[:filled_length], (-1, 1))
            else:
                x_batch[i] = np.reshape(x_seq[(random_index - self.additional_window):(random_index + self.output_length + self.additional_window)], (-1, 1))
            
        return x_batch, y_batch
            
            
                
                
            
            
                                            
        

        
