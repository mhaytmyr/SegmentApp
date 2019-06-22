#package specific imports
from modules import *

#module specific imports
from keras.engine.topology import Layer
from keras.callbacks import Callback
from keras.utils import conv_utils
from keras.engine import InputSpec

class BackPropMasker(Layer):
    def __init__(self, labels, **kwargs):
        super(BackPropMasker, self).__init__(**kwargs)
        
        # self.data_format = conv_utils.normalize_data_format(data_format)
        # self.input_spec = InputSpec(ndim=4)
        # if output_size:
        #     self.output_size = conv_utils.normalize_tuple(output_size, 2, 'output_size')
        #     self.upsampling = None
        # else:
        #     self.output_size = None
        #     self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        # if self.upsampling:
        #     height = self.upsampling[0] * input_shape[1] if input_shape[1] is not None else None
        #     width = self.upsampling[1] * input_shape[2] if input_shape[2] is not None else None
        # else:
        #     height = self.output_size[0]
        #     width = self.output_size[1]

        # return (input_shape[0],height,width,input_shape[3])
        pass

    def call(self, inputs):
        #stop gradient of the tensor if label doesn't exist
        pass


    def get_config(self):
        config = {'upsampling': self.upsampling,
                'output_size': self.output_size,
                'data_format': self.data_format}

        base_config = super(BackPropMasker, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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

