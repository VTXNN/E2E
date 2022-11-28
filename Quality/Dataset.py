import numpy as np
import pandas as pd
import gc
import datetime
from pathlib import Path
import json
import warnings
import pickle

class DataSet:
    def __init__(self, name):

        self.name = name

        self.DataFolder = 'SavedDFs'

        self.X_train = None
        self.X_test = None
        self.X_val = None

        self.y_train_z0 = None
        self.y_test_z0 = None
        self.y_val_z0 = None

        self.y_train_assoc = None
        self.y_test_assoc = None
        self.y_val_assoc = None

        self.full_feature_list = ['weight_1_0','weight_1_1','weight_1_2','weight_1_3','weight_1_4','weight_1_5','weight_1_6','weight_1_7','weight_1_8','weight_1_9',
                                  'weight_2_0','weight_2_1','weight_2_2','weight_2_3','weight_2_4','weight_2_5','weight_2_6','weight_2_7','weight_2_8','weight_2_9',
                                  'weight_3_0',
                                  'association_1_0','association_1_1','association_1_2','association_1_3','association_1_4','association_1_5','association_1_6','association_1_7','association_1_8','association_1_9',
                                  'association_1_10','association_1_11','association_1_12','association_1_13','association_1_14','association_1_15','association_1_16','association_1_17','association_1_18','association_1_19',
                                  'association_2_0','association_2_1','association_2_2','association_2_3','association_2_4','association_2_5','association_2_6','association_2_7','association_2_8','association_2_9',
                                  'association_2_10','association_2_11','association_2_12','association_2_13','association_2_14','association_2_15','association_2_16','association_2_17','association_2_18','association_2_19',
                                  'predicted_weight',
                                  'predicted_association',
                                  'trk_fromPV',
                                  'trk_z0',
                                  'predicted_z0',
                                  'true_z0',
                                  'z0_residual',
                                  'association_residual' 
                                ]

        self.z0_target_index = self.full_feature_list.index('z0_residual')
        self.assoc_target_index = self.full_feature_list.index('association_residual')

        self.training_feature_list = ['weight_1_2','weight_1_3',
                                      'weight_2_0','weight_2_1','weight_2_5','weight_2_6','weight_2_7','weight_2_9',
                                      'association_1_0','association_1_4','association_1_5','association_1_7','association_1_9',
                                      'association_1_10','association_1_12','association_1_13','association_1_14','association_1_15','association_1_16','association_1_17','association_1_18',
                                      'association_2_0','association_2_1','association_2_2','association_2_5','association_2_7',
                                      'association_2_14','association_2_15','association_2_17','association_2_18',
                                      'predicted_weight',
                                      'predicted_association',
                                      'trk_z0',
                                      'predicted_z0']

        self.training_feature_indices = [ self.full_feature_list.index(self.training_feature_list[i]) for i in range(len(self.training_feature_list))]



    def load_data_from_DF(self, filepath: str, numFiles: int, start : int = 0):
        predicted_z0 = []
        true_z0 = []
        predicted_association = []
        trk_fromPV = []

        predicted_weight = []
        trk_z0 = []

        weight_1 = { 'weight_1_'+str(i) : [] for i in range(0,10)}
        weight_2 = { 'weight_2_'+str(i) : [] for i in range(0,10)}
        weight_3_0 = []

        association_1 = { 'association_1_'+str(i) : [] for i in range(0,20)}
        association_2 = { 'association_2_'+str(i) : [] for i in range(0,20)}

        predicted_z0_residual = []
        predicted_association_residual = []

        for j in range(start,start+numFiles):
                with open(filepath+'/events_batch'+str(j)+'.pkl', 'rb') as outp:
                        Events = pickle.load(outp)
                for i,event in enumerate(Events):
                        if (i % 1000 == 0):
                                print('File: ',j,' Event: ',i, " out of ",len(Events))
                        predicted_z0.append(event['predicted_z0'])
                        true_z0.append(event['true_z0'])
                        predicted_association.append(event['predicted_association'])
                        trk_fromPV.append(event['trk_fromPV'])

                        predicted_weight.append(event['predicted_weight'])
                        trk_z0.append(event['trk_z0'])

                        [weight_1['weight_1_'+str(k)].append(event['weight_1_'+str(k)]) for k in range(0,10)]
                        [weight_2['weight_2_'+str(k)].append(event['weight_2_'+str(k)]) for k in range(0,10)]
                        weight_3_0.append(event['weight_3_0'])
                        [association_1['association_1_'+str(k)].append(event['association_1_'+str(k)]) for k in range(0,20)]
                        [association_2['association_2_'+str(k)].append(event['association_2_'+str(k)]) for k in range(0,20)]


        weight_1_array = np.array([np.concatenate(weight_1['weight_1_'+str(k)]).ravel() for k in range(0,10)])

        weight_2_array = np.array([np.concatenate(weight_2['weight_2_'+str(k)]).ravel() for k in range(0,10)])
        weight_3_0_array = np.concatenate(weight_3_0).ravel()
        association_1_array = np.array([np.concatenate(association_1['association_1_'+str(k)]).ravel() for k in range(0,20)])
        association_2_array =  np.array([np.concatenate(association_2['association_2_'+str(k)]).ravel() for k in range(0,20)])

        predicted_weight_array =  np.concatenate(predicted_weight).ravel()

        predicted_association_array =  np.concatenate(predicted_association).ravel()
        trk_fromPV_array =  np.concatenate(trk_fromPV).ravel()
        trk_z0_array = np.concatenate(trk_z0).ravel()
        predicted_z0_array =  np.concatenate(predicted_z0).ravel()
        true_z0_array =  np.concatenate(true_z0).ravel()

        z0_residual = abs(true_z0_array - predicted_z0_array)
        association_residual = abs(predicted_association_array - trk_fromPV_array)

        dataset = np.vstack([weight_1_array,
                            weight_2_array,
                            weight_3_0_array,
                            association_1_array,
                            association_2_array,
                            predicted_weight_array,
                            predicted_association_array,
                            trk_fromPV_array,
                            trk_z0_array,
                            predicted_z0_array,
                            true_z0_array,
                            z0_residual,
                            association_residual])

        print("Track Reading Complete, read: ",
                np.shape(dataset), " tracks")

        return dataset

    def generate_test_train_val(self,train_files=1,val_files=1,test_files=1):

        test = self.load_data_from_DF(self.DataFolder +'/Test',start=61,numFiles=test_files)
        train = self.load_data_from_DF(self.DataFolder +'/Train',start=0,numFiles=train_files)
        val = self.load_data_from_DF(self.DataFolder +'/Validation',start=51,numFiles=val_files)

        self.X_train = train[ self.training_feature_indices].transpose()
        self.X_test = test[ self.training_feature_indices].transpose()
        self.X_val = val[ self.training_feature_indices].transpose()

        self.y_train_z0 = train[ self.z0_target_index]
        self.y_test_z0 = test[ self.z0_target_index]
        self.y_val_z0 = val[ self.z0_target_index]

        self.y_train_assoc = train[ self.assoc_target_index]
        self.y_test_assoc = test[ self.assoc_target_index]
        self.y_val_assoc = val[ self.assoc_target_index]

        del [test,train,val]
        gc.collect()

    def save_test_train_val_h5(self, filepath):

        Path(filepath).mkdir(parents=True, exist_ok=True)

        X_train_store = pd.HDFStore(filepath+'X_train.h5')
        X_test_store = pd.HDFStore(filepath+'X_test.h5')
        X_Val_store = pd.HDFStore(filepath+'X_val.h5')

        y_train_z0_store = pd.HDFStore(filepath+'y_train_z0.h5')
        y_test_z0_store = pd.HDFStore(filepath+'y_test_z0.h5')
        y_val_z0_store = pd.HDFStore(filepath+'y_val_z0.h5')

        y_train_assoc_store = pd.HDFStore(filepath+'y_train_assoc.h5')
        y_test_assoc_store = pd.HDFStore(filepath+'y_test_assoc.h5')
        y_val_assoc_store = pd.HDFStore(filepath+'y_val_assoc.h5')

        X_train_store['df'] = self.X_train  # save it
        X_test_store['df'] = self.X_test  # save it
        y_train_store['df'] = self.y_train  # save it
        y_test_store['df'] = self.y_test  # save it

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        X_train_store.close()
        X_test_store.close()
        y_train_store.close()
        y_test_store.close()

        self.config_dict["testandtrainfilepath"] = filepath
        self.config_dict["save_timestamp"] = datetime.datetime.now().strftime(
            "%H:%M %d/%m/%y")
        with open(filepath+'config_dict.json', 'w') as f:
            json.dump(self.config_dict, f, indent=4)

        if self.verbose == 1:
            print("===Train Test Saved====")

    def load_test_train_h5(self, filepath):
        X_train_file = Path(filepath+'X_train.h5')
        if X_train_file.is_file():
            X_train_store = pd.HDFStore(filepath+'X_train.h5')
            self.X_train = X_train_store['df']
            X_train_store.close()
        else:
            print("No X Train h5 File")

        X_test_file = Path(filepath+'X_test.h5')
        if X_test_file .is_file():
            X_test_store = pd.HDFStore(filepath+'X_test.h5')
            self.X_test = X_test_store['df']
            X_test_store.close()
        else:
            print("No X Test h5 File")

        y_train_file = Path(filepath+'y_train.h5')
        if y_train_file.is_file():
            y_train_store = pd.HDFStore(filepath+'y_train.h5')
            self.y_train = y_train_store['df']
            y_train_store.close()
        else:
            print("No y train h5 file")

        y_test_file = Path(filepath+'y_test.h5')
        if y_test_file.is_file():
            y_test_store = pd.HDFStore(filepath+'y_test.h5')
            self.y_test = y_test_store['df']
            y_test_store.close()
        else:
            print("No y test h5 file")

        config_dict_file = Path(filepath+'config_dict.json')
        if config_dict_file.is_file():
            with open(filepath+'config_dict.json', 'r') as f:
                self.config_dict = json.load(f)
            self.config_dict["loaded_timestamp"] = datetime.datetime.now().strftime(
                "%H:%M %d/%m/%y")
            self.name = self.config_dict["name"]
        else:
            print("No configuration dictionary json file")


