from GBDTVertexQualityModel import XGBoostClassifierModel,SklearnClassifierModel
from Dataset import *
import numpy as np
import os
import gc

TestDataset = DataSet("z0")
TestDataset.training_feature_list = ['weight_1_2','weight_1_3',
                                     'weight_2_0','weight_2_1','weight_2_5','weight_2_6','weight_2_7','weight_2_9',
                                     'predicted_weight',
                                     'trk_z0',
                                     'predicted_z0']
TestDataset.generate_test_train_val(train_files=30,val_files=5)
TestDataset.y_train = TestDataset.y_train_z0
TestDataset.y_test = TestDataset.y_test_z0
TestDataset.y_val = TestDataset.y_val_z0


z0Model = XGBoostClassifierModel()
z0Model.comet_project_name = "vertex_z0_regressor"
z0Model.DataSet = TestDataset
z0Model.min_child_weight =  {"min":0,"max":10, "value":8.618086387114888}
z0Model.alpha            =  {"min":0,"max":1,  "value":0.17911971474844574}
z0Model.early_stopping   =  {"min":1,"max":20, "value":5}
z0Model.learning_rate    =  {"min":0,"max":1,   "value":0.8299319711448345}
z0Model.n_estimators     =  {"min":0,"max":500,   "value":106}
z0Model.subsample        =  {"min":0,"max":0.99,"value":0.12263570791260679}
z0Model.max_depth        =  {"min":1,"max":10  ,"value":4 }
z0Model.gamma            =  {"min":0,"max":0.99,"value":0.20746230788682815	}
z0Model.rate_drop        =  {"min":0,"max":1,"value":0.788588}
z0Model.skip_drop        =  {"min":0,"max":1,"value":0.147907}
#z0Model.optimise()
z0Model.train()
z0Model.test()
z0Model.evaluate('z0_model',plot=True)
z0Model.plot_model('z0_model')

del TestDataset
gc.collect()


TestDataset = DataSet("assoc")
TestDataset.training_feature_list = ['association_1_0','association_1_4','association_1_5','association_1_7','association_1_9',
                                     'association_1_10','association_1_12','association_1_13','association_1_14','association_1_15','association_1_16','association_1_17','association_1_18',
                                     'association_2_0','association_2_1','association_2_2','association_2_5','association_2_7',
                                     'association_2_14','association_2_15','association_2_17','association_2_18',
                                     'predicted_weight',
                                     'predicted_association',
                                     'trk_z0',
                                     'predicted_z0']
TestDataset.generate_test_train_val(train_files=30,val_files=5)
TestDataset.y_train = TestDataset.y_train_assoc
TestDataset.y_test = TestDataset.y_test_assoc
TestDataset.y_val = TestDataset.y_val_assoc
assocModel = XGBoostClassifierModel()
assocModel.comet_project_name = "vertex_assoc_regressor"
assocModel.DataSet = TestDataset
assocModel.optimise()
assocModel.train()
assocModel.test()
assocModel.evaluate('assoc_model',plot=True)
assocModel.plot_model('Assoc_model')
