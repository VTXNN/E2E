from comet_ml import Optimizer
from Dataset import DataSet
from sklearn import metrics
import numpy as np
import joblib
from pathlib import Path
import os
import xgboost as xgb 
#import tensorflow_decision_forests as tfdf
from textwrap import wrap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import mplhep as hep
#hep.set_style("CMSTex")
hep.cms.label()
hep.cms.text("Simulation")
plt.style.use(hep.style.CMS)

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

LEGEND_WIDTH = 20
LINEWIDTH = 3
MARKERSIZE = 20

nbins = 256

colormap = "jet"

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('axes', linewidth=5)              # thickness of axes
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)            # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams['xtick.major.size'] = 20
matplotlib.rcParams['xtick.major.width'] = 5
matplotlib.rcParams['xtick.minor.size'] = 10
matplotlib.rcParams['xtick.minor.width'] = 4

matplotlib.rcParams['ytick.major.size'] = 20
matplotlib.rcParams['ytick.major.width'] = 5
matplotlib.rcParams['ytick.minor.size'] = 10
matplotlib.rcParams['ytick.minor.width'] = 4

colours=["red","green","blue","orange","purple","yellow"]

class TrackClassifierModel:
    def __init__(self):
        self.DataSet = None
        
        self.model = None
        self.y_predict = None
        self.y_predict_proba = None

        self.synth_y_predict = None
        self.synth_y_predict_proba = None

        self.z0 = False

        self.comet_api_key = "expKifKow3Mn4dnjc1UGGOqrg"
        self.comet_project_name = "z0_regressor_test"
    
    @classmethod
    def load_from_pkl(cls,model_filename):
        pklclass = cls()
        pklclass.model = joblib.load(model_filename+".pkl")
        return pklclass

    def load_data(self,filepath):
        self.DataSet = DataSet.fromTrainTest(filepath)
        print(self.DataSet)
    def train(self):
        pass
    def test(self):
        pass
    def optimise(self):
        pass
    def evaluate(self,name="model",plot=False,val=False):
        if not val:
            MSE = metrics.mean_squared_error(self.DataSet.y_test,self.y_predict)
            MAE = metrics.mean_absolute_error(self.DataSet.y_test,self.y_predict)
        else:
            MSE = metrics.mean_squared_error(self.DataSet.y_val,self.y_predict)
            MAE = metrics.mean_absolute_error(self.DataSet.y_val,self.y_predict)

        print(str(name) + " MSE: ",MSE)
        print(str(name) + " MAE: ",MAE)

        if plot:
            self.residual(name)

        return MSE,MAE

    def residual(self,name="model"):
        plt.clf()
        fig,ax = plt.subplots(1,2,figsize=(20,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])

        if self.z0:
            bin_range=(-15,15)
            log_bin_range=(-1,1)
            axis_label = 'z error Residual [cm]'

        else:
            bin_range=(-5,5)
            log_bin_range=(-1,1)
            axis_label = 'Assoc error Residual'
            
            
        ax[0].hist(self.DataSet.y_test-self.y_predict,bins=50,range=bin_range,histtype="step",
                        linewidth=LINEWIDTH,color = colours[0],
                        label='\n'.join(wrap(f"%s \nMSE = %.4f" 
                        %(name,metrics.mean_squared_error(self.DataSet.y_test,self.y_predict)),LEGEND_WIDTH)))
            
        quartiles = np.percentile(self.DataSet.y_test-self.y_predict,[32,50,68])

        ax[1].hist(self.DataSet.y_test-self.y_predict,bins=50,range=log_bin_range,histtype="step",
                        linewidth=LINEWIDTH,color = colours[0],
                        label='\n'.join(wrap(f"%s \nQuartile Width = %.4f" 
                        %(name,quartiles[2]-quartiles[0]),LEGEND_WIDTH)))


            
        ax[0].grid(True)
        ax[0].set_xlabel(axis_label,ha="right",x=1)
        ax[0].set_ylabel('Tracks',ha="right",y=1)
        ax[0].set_yscale("log")
        ax[0].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

        ax[1].grid(True)
        ax[1].set_xlabel(axis_label,ha="right",x=1)
        ax[1].set_ylabel('Tracks',ha="right",y=1)
        ax[1].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

        plt.tight_layout()
        plt.savefig(name+'_residual.png',dpi=600)

    def save_model(self,filepath):
        joblib.dump(self.model,filepath + ".pkl")
    def synth_model(self):
        pass

class GBDTClassifierModel(TrackClassifierModel):
    def __init__(self):
        super().__init__()

        self.learning_rate =  {"min":0,"max":1,   "value":0.1}
        self.n_estimators  =  {"min":0,"max":150,"value":100}
        self.subsample     =  {"min":0,"max":0.99,"value":0.5}
        self.max_depth     =  {"min":1,"max":5  ,"value":3  }
        self.gamma         =  {"min":0,"max":0.99,"value":0.01}
        
        self.model = None 

    def train(self):
        self.model = self.model.fit(self.DataSet.X_train,self.DataSet.y_train)

    def test(self,val=False):
        if val:
            self.y_predict = self.model.predict(self.DataSet.X_val)
        else:
            self.y_predict = self.model.predict(self.DataSet.X_test)

    def save_model(self,filepath):
        joblib.dump(self.model,filepath + ".pkl")

    def optimise(self):
        pass

    def synth_model(self):
        pass

class SklearnClassifierModel(GBDTClassifierModel):
    def __init__(self):
        from sklearn.ensemble import GradientBoostingRegressor
        super().__init__()

        self.model = GradientBoostingRegressor(loss="squared_error",
                                                learning_rate = self.learning_rate["value"],
                                                n_estimators  = self.n_estimators["value"],
                                                subsample     = self.subsample["value"],
                                                max_depth     = self.max_depth["value"],
                                                ccp_alpha     = self.gamma["value"],
                                                verbose       = 2,
                                                )

    def optimise(self):
        config = {
                    # We pick the Bayes algorithm:
                    "algorithm": "bayes",

                    # Declare your hyperparameters in the Vizier-inspired format:
                    "parameters": {
                        "n_estimators":  {"type": "int", "min": self.n_estimators["min"], 
                                          "max": self.n_estimators["max"],  "scalingType": "uniform"},
                        "max_depth":     {"type": "int", "min": self.max_depth["min"], 
                                          "max": self.max_depth["max"],     "scalingType": "uniform"},
                        "learning_rate": {"type": "float", "min": self.learning_rate["min"], 
                                          "max": self.learning_rate["max"], "scalingType": "uniform"},
                        "gamma":         {"type": "float", "min": self.gamma["min"], 
                                          "max": self.gamma["max"] ,        "scalingType": "uniform"},
                        "subsample":     {"type": "float", "min": self.subsample["min"], 
                                          "max": self.subsample["max"],     "scalingType": "uniform"},
                    },

                    # Declare what we will be optimizing, and how:
                    "spec": {
                    "metric": "MSE",
                        "objective": "minimize",
                    },
                }
        opt = Optimizer(config, api_key=self.comet_api_key, project_name=self.comet_project_name,auto_metric_logging=True)

        for experiment in opt.get_experiments():
            self.model.learning_rate = experiment.get_parameter("learning_rate")
            self.model.n_estimators = experiment.get_parameter("n_estimators")
            self.model.subsample = experiment.get_parameter("subsample")
            self.model.max_depth = experiment.get_parameter("max_depth")
            self.model.ccp_alpha = experiment.get_parameter("gamma")

            self.train()
            self.test()
            auc,binary_accuracy = self.evaluate()

            experiment.log_metric("ROC",auc)
            experiment.log_metric("Binary_Accuracy",binary_accuracy)

    def synth_model(self,sim : bool = True,hdl : bool = True,hls : bool = True,test_events : int = 1000):
        import shutil

        if sim:
            if Path("simdir").is_dir():
              shutil.rmtree("simdir")

            from scipy.special import expit
            from conifer import conifer
            simcfg = conifer.backends.vhdl.auto_config()
            simcfg['Precision'] = 'ap_fixed<16,8>'
            # Set the output directory to something unique
            simcfg['OutputDir'] = "simdir/"
            simcfg["XilinxPart"] = 'xcvu9p-flga2104-2L-e'
            simcfg["ClockPeriod"] = 2.78

            simmodel = conifer.model(self.model, conifer.converters.sklearn, conifer.backends.vhdl, simcfg)
            simmodel.compile()

            # Run HLS C Simulation and get the output
            if test_events == 0:
              length = -1
            else:
              length = test_events
            
            temp_decision = simmodel.decision_function(self.DataSet.X_test[0:length])
            self.synth_y_predict_proba = expit(temp_decision)
            temp_array = np.empty_like(temp_decision)
            temp_array[self.synth_y_predict_proba > 0.5] = 1
            temp_array[self.synth_y_predict_proba <= 0.5] = 0
            self.synth_y_predict = temp_array
            

            print("AUC ROC sim: {}".format(metrics.roc_auc_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)], self.synth_y_predict_proba)))
            print("AUC ROC sklearn: {}".format(metrics.roc_auc_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)], self.y_predict_proba[0:len(self.synth_y_predict_proba)])))
            print("Accuracy sim: {}".format(metrics.accuracy_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)],  self.synth_y_predict)))
            print("Accuracy sklearn: {}".format(metrics.accuracy_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)],  self.y_predict[0:len(self.synth_y_predict_proba)])))

        if hdl:
            if Path("hdldir").is_dir():
                shutil.rmtree("hdldir")

            hdlcfg = conifer.backends.vhdl.auto_config()
            hdlcfg['Precision'] = 'ap_fixed<16,8>'
            # Set the output directory to something unique
            hdlcfg['OutputDir'] = "hdldir/"
            hdlcfg["XilinxPart"] = 'xcvu9p-flga2104-2L-e'
            hdlcfg["ClockPeriod"] = 2.78
            hdlmodel = conifer.model(self.model, conifer.converters.sklearn, conifer.backends.vhdl, hdlcfg)
            hdlmodel.write()
            hdlmodel.build()
        if hls:
            if Path("hlsdir").is_dir():
                shutil.rmtree("hlsdir")

            hlscfg = conifer.backends.vivadohls.auto_config()
            hlscfg['Precision'] = 'ap_fixed<16,8>'
            # Set the output directory to something unique
            hlscfg['OutputDir'] = "hlsdir/"
            hlscfg["XilinxPart"] = 'xcvu9p-flga2104-2L-e'
            hlscfg["ClockPeriod"] = 2.78

            # Create and compile the model
            hlsmodel = conifer.model(self.model, conifer.converters.sklearn, conifer.backends.vivadohls, hlscfg)
            hlsmodel.write()
            hlsmodel.build()

class XGBoostClassifierModel(GBDTClassifierModel):
    def __init__(self):
        super().__init__()

        self.min_child_weight =  {"min":0,"max":10, "value":1.5176133095340514}
        self.alpha            =  {"min":0,"max":1,  "value":0.6274160352470041}
        self.early_stopping   =  {"min":1,"max":20, "value":5}
        self.learning_rate    =  {"min":0,"max":1,   "value":0.8299319711448345}
        self.n_estimators     =  {"min":0,"max":500,   "value":3}
        self.subsample        =  {"min":0,"max":0.99,"value":0.4742430167523669}
        self.max_depth        =  {"min":1,"max":10  ,"value":5 }
        self.gamma            =  {"min":0,"max":0.99,"value":0.2829920211730428	}
        self.rate_drop        =  {"min":0,"max":1,"value":0.788588}
        self.skip_drop        =  {"min":0,"max":1,"value":0.147907}



        self.model = xgb.XGBRegressor(n_estimators      = self.n_estimators["value"],
                                       max_depth         = self.max_depth["value"],
                                       learning_rate     = self.learning_rate["value"],
                                       gamma             = self.gamma["value"],
                                       min_child_weight  = self.min_child_weight["value"],
                                       subsample         = self.subsample["value"],
                                       reg_alpha         = self.alpha["value"] ,
                                       objective         = 'reg:squarederror',
                                       #booster           = "dart",
                                       #rate_drop         = self.rate_drop["value"] ,
                                       #skip_drop         = self.skip_drop["value"] ,
                                       tree_method       = 'exact',
                                       n_jobs            = 8)

    def optimise(self):
        config = {
                    # We pick the Bayes algorithm:
                    "algorithm": "bayes",

                    # Declare your hyperparameters in the Vizier-inspired format:
                    "parameters": {
                        "n_estimators":     {"type": "integer", "min": self.n_estimators["min"], 
                                             "max": self.n_estimators["max"],     "scalingType": "uniform"},
                        "max_depth":        {"type": "integer", "min": self.max_depth["min"], 
                                             "max": self.max_depth["max"],        "scalingType": "uniform"},
                        "learning_rate":    {"type": "float", "min": self.learning_rate["min"], 
                                             "max": self.learning_rate["max"],    "scalingType": "uniform"},
                        "gamma":            {"type": "float", "min": self.gamma["min"], 
                                             "max": self.gamma["max"] ,           "scalingType": "uniform"},
                        "subsample":        {"type": "float", "min": self.subsample["min"], 
                                             "max": self.subsample["max"],        "scalingType": "uniform"},
                        "min_child_weight": {"type": "float", "min": self.min_child_weight["min"], 
                                             "max": self.min_child_weight["max"], "scalingType": "uniform"},
                        "reg_alpha":        {"type": "float", "min": self.alpha["min"], 
                                             "max": self.alpha["max"],            "scalingType": "uniform"},
                    },

                    # Declare what we will be optimizing, and how:
                    "spec": {
                    "metric": "MSE",
                        "objective": "minimize",
                    },
                }
        opt = Optimizer(config, api_key=self.comet_api_key, project_name=self.comet_project_name,auto_metric_logging=True)

        for experiment in opt.get_experiments():
            self.model.learning_rate = experiment.get_parameter("learning_rate")
            self.model.n_estimators = experiment.get_parameter("n_estimators")
            self.model.subsample = experiment.get_parameter("subsample")
            self.model.max_depth = experiment.get_parameter("max_depth")
            self.model.gamma = experiment.get_parameter("gamma")
            self.model.reg_alpha =experiment.get_parameter("reg_alpha")
            self.model.min_child_weight = experiment.get_parameter("min_child_weight")

            self.train()
            self.test(val=True)
            MSE,MAE = self.evaluate(val=True)

            experiment.log_metric("MSE",MSE)
            experiment.log_metric("MAE",MAE)

    def load_model(self,filepath):
        self.model = joblib.load(filepath + ".pkl")

    def plot_model(self,name):
        import matplotlib.pyplot as plt
        xgb.plot_importance(self.model)
        plt.savefig(name+"_feat_importance.png")

    def synth_model(self,sim : bool = True,hdl : bool = True,hls : bool = True,test_events : int = 1000,intwidth : int = 32,fracwidth : int = 16, plot : bool = False):
        import shutil

        if sim:
            if Path("simdir").is_dir():
              shutil.rmtree("simdir")

            from scipy.special import expit
            import conifer
            simcfg = conifer.backends.vhdl.auto_config()
            simcfg['Precision'] = 'ap_fixed<'+str(intwidth)+','+str(fracwidth)+'>'
            # Set the output directory to something unique
            simcfg['OutputDir'] = "simdir/"
            simcfg["XilinxPart"] = 'xcvu9p-flga2104-2L-e'
            simcfg["ClockPeriod"] = 2.78

            self.simmodel = conifer.model(self.model.get_booster(), conifer.converters.xgboost, conifer.backends.vhdl, simcfg)
            self.simmodel.compile()

            # Run HLS C Simulation and get the output

            if test_events > 0:

                length = test_events
    
                temp_decision = self.simmodel.decision_function(self.DataSet.X_test[0:length])
                self.synth_y_predict_proba = expit(temp_decision)
                temp_array = np.empty_like(temp_decision)
                temp_array[self.synth_y_predict_proba > 0.5] = 1
                temp_array[self.synth_y_predict_proba <= 0.5] = 0
                self.synth_y_predict = temp_array

                simauc = metrics.roc_auc_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)], self.synth_y_predict_proba)
                xgbsim = metrics.roc_auc_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)], self.y_predict_proba[0:len(self.synth_y_predict_proba)])
                

                print("AUC ROC sim: {}".format(simauc))
                print("AUC ROC xgb: {}".format(xgbsim))
                print("Accuracy sim: {}".format(metrics.accuracy_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)],  self.synth_y_predict)))
                print("Accuracy sklearn: {}".format(metrics.accuracy_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)],  self.y_predict[0:len(self.synth_y_predict_proba)])))
        
                if plot:
                    fig, ax = plt.subplots(1,1, figsize=(18,18)) 
                    ax.tick_params(axis='x', labelsize=16)
                    ax.tick_params(axis='y', labelsize=16)
                    ax.tick_params(axis='x', labelsize=16)
                    ax.tick_params(axis='y', labelsize=16)
                    fprsim, tprsim, thresholdssim = metrics.roc_curve(self.DataSet.y_test[0:len(self.synth_y_predict_proba)], self.synth_y_predict_proba, pos_label=1)
                    fprxgb, tprxgb, thresholdsxgb = metrics.roc_curve(self.DataSet.y_test[0:len(self.synth_y_predict_proba)], self.y_predict_proba[0:len(self.synth_y_predict_proba)], pos_label=1)
                    ax.set_title("ROC Curve",loc='left',fontsize=20)
                    ax.plot(fprsim,tprsim,label="Conifer Sim <"+str(intwidth)+","+str(fracwidth)+">" + "AUC: %.3f"%simauc)
                    ax.plot(fprxgb,tprxgb,label="XGBoost AUC: %.3f"%simauc)
                    ax.set_xlim([0,0.3])
                    ax.set_ylim([0.8,1.0])
                    ax.set_xlabel("Fake Positive Rate",ha="right",x=1,fontsize=16)
                    ax.set_ylabel("Identification Efficiency",ha="right",y=1,fontsize=16)
                    ax.legend()
                    ax.grid()
                    plt.savefig("SimAUC.png",dpi=600)

        if hdl:
            if Path("hdldir").is_dir():
                shutil.rmtree("hdldir")

            hdlcfg = conifer.backends.vhdl.auto_config()
            hdlcfg['Precision'] = 'ap_fixed<32,16>'
            # Set the output directory to something unique
            hdlcfg['OutputDir'] = "hdldir/"
            hdlcfg["XilinxPart"] = 'xcvu9p-flga2104-2L-e'
            hdlcfg["ClockPeriod"] = 2.78
            hdlmodel = conifer.model(self.model.get_booster(), conifer.converters.xgboost, conifer.backends.vhdl, hdlcfg)
            hdlmodel.write()
            hdlmodel.build()
        if hls:
            if Path("hlsdir").is_dir():
                shutil.rmtree("hlsdir")

            hlscfg = conifer.backends.vivadohls.auto_config()
            hlscfg['Precision'] = 'ap_fixed<32,16>'
            # Set the output directory to something unique
            hlscfg['OutputDir'] = "hlsdir/"
            hlscfg["XilinxPart"] = 'xcvu9p-flga2104-2L-e'
            hlscfg["ClockPeriod"] = 2.78

            # Create and compile the model
            hlsmodel = conifer.model(self.model.get_booster(), conifer.converters.xgboost, conifer.backends.vivadohls, hlscfg)
            hlsmodel.write()
            hlsmodel.build()

    def ONNX_convert_model(self,filepath):
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        num_features = len(self.DataSet.config_dict["trainingFeatures"])
        X = np.array(np.random.rand(10, num_features), dtype=np.float32)
        
        print(self.model.predict(X))

        # The name of the input is needed in Clasifier_cff as GBDTIdONNXInputName
        initial_type = [('feature_input', FloatTensorType([1, num_features]))]

        
        onx = onnxmltools.convert.convert_xgboost(self.model, initial_types=initial_type)

        # Save the model
        with open(filepath+".onnx", "wb") as f:
          f.write(onx.SerializeToString())

        # This tests the model
        import onnxruntime as rt

        # setup runtime - load the persisted ONNX model
        sess = rt.InferenceSession(filepath+".onnx")

        # get model metadata to enable mapping of new input to the runtime model.
        input_name = sess.get_inputs()[0].name
        # This label will access the class probabilities when run in CMSSW, use index 0 for class prediction
        label_name = sess.get_outputs()[1].name


        print(sess.get_inputs()[0].name)
        # The name of the output is needed in Clasifier_cff as GBDTIdONNXOutputName
        print(label_name)

        # predict on random input and compare to previous XGBoost model
        for i in range(len(X)):
          pred_onx = sess.run([], {input_name: X[i:i+1]})[1]
        print(pred_onx)

class FullXGBoostClassifierModel(XGBoostClassifierModel):
    def __init__(self):
        super().__init__()

        self.min_child_weight =  {"min":0,"max":10, "value":8.582127}
        self.alpha            =  {"min":0,"max":1,  "value":0.54012275}
        self.early_stopping   =  {"min":1,"max":20, "value":5}
        self.learning_rate    =  {"min":0,"max":1,   "value":0.6456304}
        self.n_estimators     =  100
        self.subsample        =  {"min":0,"max":0.99,"value":0.343124}
        self.max_depth        =  {"min":1,"max":3  ,"value":3  }
        self.gamma            =  {"min":0,"max":0.99,"value":0.314177}
        self.rate_drop        =  {"min":0,"max":1,"value":0.788588}
        self.skip_drop        =  {"min":0,"max":1,"value":0.147907}

        self.dtest = None
        self.dtrain = None

        self.comet_project_name = "Xgboost_test"

        self.param = {"max_depth":self.max_depth["value"],
                      "learning_rate":self.learning_rate["value"],
                      "min_child_weight":self.min_child_weight["value"],
                      "gamma":self.gamma["value"],
                      'eval_metric':'auc',
                      "subsample":self.subsample["value"],
                      "reg_alpha":self.alpha["value"] ,
                      "objective":'binary:logistic',
                      "nthread":8,
                      "booster":"dart",
                      "rate_drop":self.rate_drop["value"] ,
                      "skip_drop":self.skip_drop["value"]}
        self.num_rounds = self.n_estimators
        self.early_stopping_rounds = self.early_stopping["value"]


        self.model = None

    def load_data(self,filepath):
        self.DataSet = DataSet.fromTrainTest(filepath)
        self.dtrain = xgb.DMatrix(self.DataSet.X_train.to_numpy(),label=np.ravel(self.DataSet.y_train))
        self.dtest = xgb.DMatrix(self.DataSet.X_test.to_numpy(),label=np.ravel(self.DataSet.y_test))

    def train(self):
        
        cv = xgb.cv(self.param, self.dtrain, self.num_rounds, nfold=5,
               metrics={'auc'}, seed=4,
               callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True),xgb.callback.EarlyStopping(self.early_stopping_rounds)])

        self.boost_rounds = cv['test-auc-mean'].argmax()
        self.model = xgb.train(self.param,self.dtrain,num_boost_round=self.boost_rounds)

    def test(self):
        
        self.y_predict_proba = self.model.predict(self.dtest)
        self.y_predict = np.heaviside(self.y_predict_proba -0.5,0)


    def optimise(self):
        config = {
                    # We pick the Bayes algorithm:
                    "algorithm": "bayes",

                    # Declare your hyperparameters in the Vizier-inspired format:
                    "parameters": {
                        "max_depth":        {"type": "integer", "min": self.max_depth["min"], 
                                             "max": self.max_depth["max"],        "scalingType": "uniform"},
                        "learning_rate":    {"type": "float", "min": self.learning_rate["min"], 
                                             "max": self.learning_rate["max"],    "scalingType": "uniform"},
                        "gamma":            {"type": "float", "min": self.gamma["min"], 
                                             "max": self.gamma["max"] ,           "scalingType": "uniform"},
                        "subsample":        {"type": "float", "min": self.subsample["min"], 
                                             "max": self.subsample["max"],        "scalingType": "uniform"},
                        "min_child_weight": {"type": "float", "min": self.min_child_weight["min"], 
                                             "max": self.min_child_weight["max"], "scalingType": "uniform"},
                        "reg_alpha":        {"type": "float", "min": self.alpha["min"], 
                                             "max": self.alpha["max"],            "scalingType": "uniform"},
                        "early_stopping":   {"type": "integer", "min": self.early_stopping["min"], 
                                             "max": self.early_stopping["max"],     "scalingType": "uniform"},
                        "rate_drop":        {"type": "float", "min": self.rate_drop["min"], 
                                             "max": self.rate_drop["max"],     "scalingType": "uniform"},
                        "skip_drop":        {"type": "float", "min": self.skip_drop["min"], 
                                             "max": self.skip_drop["max"],     "scalingType": "uniform"}
                    },

                    # Declare what we will be optimizing, and how:
                    "spec": {
                    "metric": "ROC",
                        "objective": "maximize",
                    },
                }
        opt = Optimizer(config, api_key=self.comet_api_key, project_name=self.comet_project_name,auto_metric_logging=True)

        for experiment in opt.get_experiments():
            self.param["learning_rate"] = experiment.get_parameter("learning_rate")
            self.num_rounds = self.n_estimators
            self.param["subsample"]= experiment.get_parameter("subsample")
            self.param["max_depth"] = experiment.get_parameter("max_depth")
            self.param["gamma"] = experiment.get_parameter("gamma")
            self.param["reg_alpha"] =experiment.get_parameter("reg_alpha")
            self.param["min_child_weight"] = experiment.get_parameter("min_child_weight")
            self.param["rate_drop"] = experiment.get_parameter("rate_drop")
            self.param["skip_drop"] = experiment.get_parameter("skip_drop")
            self.early_stopping_rounds =  experiment.get_parameter("early_stopping")

            self.train()
            self.test()
            auc,binary_accuracy = self.evaluate()

            experiment.log_metric("ROC",auc)
            experiment.log_metric("Binary_Accuracy",binary_accuracy)
            experiment.log_metric("Best Boost Round",self.boost_rounds)
            experiment.log_metric("score",(auc/0.5-1)+(1-self.boost_rounds/self.n_estimators ))

    def plot_model(self):
        xgb.plot_importance(self.model)

    def synth_model(self,sim : bool = True,hdl : bool = True,hls : bool = True,test_events : int = 1000):
        import shutil

        if sim:
            if Path("simdir").is_dir():
              shutil.rmtree("simdir")

            from scipy.special import expit
            from conifer import conifer
            simcfg = conifer.backends.vhdl.auto_config()
            simcfg['Precision'] = 'ap_fixed<13,6>'
            # Set the output directory to something unique
            simcfg['OutputDir'] = "simdir/"
            simcfg["XilinxPart"] = 'xcvu9p-flga2104-2L-e'
            simcfg["ClockPeriod"] = 2.78

            self.simmodel = conifer.model(self.model, conifer.converters.xgboost, conifer.backends.vhdl, simcfg)
            self.simmodel.compile()

            # Run HLS C Simulation and get the output
            

            if test_events > 0:
                length = test_events
            
                temp_decision = self.simmodel.decision_function(self.DataSet.X_test[0:length])
                self.synth_y_predict_proba = expit(temp_decision)
                temp_array = np.empty_like(temp_decision)
                temp_array[self.synth_y_predict_proba > 0.5] = 1
                temp_array[self.synth_y_predict_proba <= 0.5] = 0
                self.synth_y_predict = temp_array
                

                print("AUC ROC sim: {}".format(metrics.roc_auc_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)], self.synth_y_predict_proba)))
                print("AUC ROC sklearn: {}".format(metrics.roc_auc_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)], self.y_predict_proba[0:len(self.synth_y_predict_proba)])))
                print("Accuracy sim: {}".format(metrics.accuracy_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)],  self.synth_y_predict)))
                print("Accuracy sklearn: {}".format(metrics.accuracy_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)],  self.y_predict[0:len(self.synth_y_predict_proba)])))


        if hdl:
            if Path("hdldir").is_dir():
                shutil.rmtree("hdldir")

            hdlcfg = conifer.backends.vhdl.auto_config()
            hdlcfg['Precision'] = 'ap_fixed<13,6>'
            # Set the output directory to something unique
            hdlcfg['OutputDir'] = "hdldir/"
            hdlcfg["XilinxPart"] = 'xcvu9p-flga2104-2L-e'
            hdlcfg["ClockPeriod"] = 2.78
            hdlmodel = conifer.model(self.model, conifer.converters.xgboost, conifer.backends.vhdl, hdlcfg)
            hdlmodel.write()
            hdlmodel.build()
        if hls:
            if Path("hlsdir").is_dir():
                shutil.rmtree("hlsdir")

            hlscfg = conifer.backends.vivadohls.auto_config()
            hlscfg['Precision'] = 'ap_fixed<13,6>'
            # Set the output directory to something unique
            hlscfg['OutputDir'] = "hlsdir/"
            hlscfg["XilinxPart"] = 'xcvu9p-flga2104-2L-e'
            hlscfg["ClockPeriod"] = 2.78

            # Create and compile the model
            hlsmodel = conifer.model(self.model, conifer.converters.xgboost, conifer.backends.vivadohls, hlscfg)
            hlsmodel.write()
            hlsmodel.build()

class TFDFClassifierModel(GBDTClassifierModel):
    def __init__(self):
        super().__init__()

        

        self.min_child_weight =  {"min":0,"max":10, "value":1.3690275705621135}
        self.alpha            =  {"min":0,"max":1,  "value":0.9307933560230425}
        self.early_stopping   =  {"min":1,"max":20, "value":5}
        self.learning_rate    =  {"min":0,"max":1,   "value":0.3245287291246959}
        self.n_estimators     =  {"min":0,"max":250,   "value":100}
        self.subsample        =  {"min":0,"max":0.99,"value":0.2459092973919883	}
        self.max_depth        =  {"min":1,"max":5  ,"value":3 }
        self.gamma            =  {"min":0,"max":0.99,"value":0.9730465449052166	}
        self.rate_drop        =  {"min":0,"max":1,"value":0.788588}
        self.skip_drop        =  {"min":0,"max":1,"value":0.147907}


        self.model = tfdf.keras.RandomForestModel()




        self.model = xgb.XGBClassifier(n_estimators      = self.n_estimators["value"],
                                       max_depth         = self.max_depth["value"],
                                       learning_rate     = self.learning_rate["value"],
                                       gamma             = self.gamma["value"],
                                       min_child_weight  = self.min_child_weight["value"],
                                       subsample         = self.subsample["value"],
                                       reg_alpha         = self.alpha["value"] ,
                                       objective         = 'binary:logistic',
                                       #booster           = "dart",
                                       #rate_drop         = self.rate_drop["value"] ,
                                       #skip_drop         = self.skip_drop["value"] ,
                                       tree_method       = 'exact',
                                       use_label_encoder = False ,
                                       n_jobs            = 8)

    def optimise(self):
        config = {
                    # We pick the Bayes algorithm:
                    "algorithm": "bayes",

                    # Declare your hyperparameters in the Vizier-inspired format:
                    "parameters": {
                        "n_estimators":     {"type": "integer", "min": self.n_estimators["min"], 
                                             "max": self.n_estimators["max"],     "scalingType": "uniform"},
                        "max_depth":        {"type": "integer", "min": self.max_depth["min"], 
                                             "max": self.max_depth["max"],        "scalingType": "uniform"},
                        "learning_rate":    {"type": "float", "min": self.learning_rate["min"], 
                                             "max": self.learning_rate["max"],    "scalingType": "uniform"},
                        "gamma":            {"type": "float", "min": self.gamma["min"], 
                                             "max": self.gamma["max"] ,           "scalingType": "uniform"},
                        "subsample":        {"type": "float", "min": self.subsample["min"], 
                                             "max": self.subsample["max"],        "scalingType": "uniform"},
                        "min_child_weight": {"type": "float", "min": self.min_child_weight["min"], 
                                             "max": self.min_child_weight["max"], "scalingType": "uniform"},
                        "reg_alpha":        {"type": "float", "min": self.alpha["min"], 
                                             "max": self.alpha["max"],            "scalingType": "uniform"},
                    },

                    # Declare what we will be optimizing, and how:
                    "spec": {
                    "metric": "ROC",
                        "objective": "maximize",
                    },
                }
        opt = Optimizer(config, api_key=self.comet_api_key, project_name=self.comet_project_name,auto_metric_logging=True)

        for experiment in opt.get_experiments():
            self.model.learning_rate = experiment.get_parameter("learning_rate")
            self.model.n_estimators = experiment.get_parameter("n_estimators")
            self.model.subsample = experiment.get_parameter("subsample")
            self.model.max_depth = experiment.get_parameter("max_depth")
            self.model.gamma = experiment.get_parameter("gamma")
            self.model.reg_alpha =experiment.get_parameter("reg_alpha")
            self.model.min_child_weight = experiment.get_parameter("min_child_weight")

            self.train()
            self.test()
            auc,binary_accuracy = self.evaluate()

            experiment.log_metric("ROC",auc)
            experiment.log_metric("Binary_Accuracy",binary_accuracy)

    def save_model(self,filepath):
        self.model.save(filepath)

    def load_model(self,filepath):
        self.model = joblib.load(filepath + ".pkl")

    def load_data(self,filepath):
        self.DataSet = DataSet.fromTrainTest(filepath)
        self.train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(self.DataSet.X_train, label="trk_fake")
        self.test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(self.DataSet.X_test, label="trk_fake")

    def train(self):
        self.model.compile(metrics=["binary_crossentropy"])
        with sys_pipes():
          self.model.fit(x=self.train_ds)

        print(self.model.summary())
        print(self.model.make_inspector().variable_importances())


    def test(self):
        evaluation = self.model_1.evaluate(self.test_ds, return_dict=True)
        for name, value in evaluation.items():
            print(f"{name}: {value:.4f}")

    def plot_model(self):
        tfdf.model_plotter.plot_model_in_colab(self.model, tree_idx=0, max_depth=3)

