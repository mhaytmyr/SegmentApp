#package specific imports
from modules import *

#module specific imports
from keras.engine.topology import Layer
from keras.callbacks import Callback
from keras.utils import conv_utils
from keras.engine import InputSpec


class Metrics(Callback):
    def __init__(self,generator, steps, batch):
        self.generator = generator;
        self.steps = steps;
        self.batch = batch;

    def on_train_begin(self,logs={}):
        n = self.params["metrics"].__len__();
        self.val_score_names = self.params["metrics"][n//2:];
        self.train_score_names = self.params["metrics"][:n//2];
        
        self.train_metrics = {item:0 for item in self.train_score_names if 'loss' not in item};
        self.batch_accumilator = {item:0 for item in self.train_score_names if 'loss' not in item};

        print(self.val_score_names)
        print(self.train_score_names)

    def on_epoch_begin(self,epoch,logs={}):
        #reset values to zero
        for key in self.train_metrics:
            self.train_metrics[key] = 0;
            self.batch_accumilator[key] = 0;

    def on_batch_end(self,batch,logs={}):

        #now perform weighets sum
        for key in self.train_metrics:
            #don't update results that are zero
            if logs[key]>0:
                #update batch accumilator
                self.batch_accumilator[key] += logs["size"]; 
                self.train_metrics[key] += logs["size"]*logs[key];
            
    def on_epoch_end(self,epoch,logs={}):

        print("end of epochs");
        print("original scores ....")
        #print(logs)
        self.train_scores = {key:logs[key] for key in self.train_score_names if 'loss' not in key};
        print(self.train_scores)
        
        print("manual calculation ....")        
        self.train_scores = {key:self.train_metrics[key]/max(1,self.batch_accumilator[key]) for key in self.train_metrics};
        print(self.train_scores)    

        #assign updated scores to log
        for key in self.train_scores:
            logs[key] = self.train_scores[key]

