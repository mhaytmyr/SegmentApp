#package specific imports
from modules import *

#module specific imports
import keras as K
import tensorflow as tf
from keras.engine.topology import get_source_inputs
from keras.applications import imagenet_utils
from keras.utils import conv_utils
from keras.utils.data_utils import get_file

from keras.backend import tf as ktf
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, AveragePooling2D
from keras.layers import SeparableConv2D, MaxPooling2D, UpSampling2D, Conv2D, Dropout
from keras.layers import Lambda, Input, BatchNormalization, Concatenate, ZeroPadding2D, Add, Multiply
from keras.models import Model, model_from_json
from keras.layers.advanced_activations import PReLU, LeakyReLU

from modules.custom_layers import Metrics

class SegTHOR:
    def __init__(self,numLabels):
        #number of output labels
        self.numLabels = numLabels

    @staticmethod
    def metric_per_label(y_true, y_pred, alpha = 0.5, beta = 0.5):

        true_positives = y_true * y_pred
        false_negatives = y_true * (1 - y_pred)
        false_positives = (1 - y_true) * y_pred

        num = K.backend.sum(true_positives, axis = (0,1,2)) #compute loss per-batch
        den = num+alpha*K.backend.sum(false_negatives, axis = (0,1,2)) + beta*K.backend.sum(false_positives, axis=(0,1,2))+1
        T = K.backend.mean(num/den)

        return T

    def get_label_dice(self,label):
    
        def dice(y_true,y_pred):
            y_true = K.backend.argmax(y_true,axis=-1)
            y_pred = K.backend.argmax(y_pred,axis=-1)

            true = K.backend.cast(K.backend.equal(y_true,label),'float32')
            pred = K.backend.cast(K.backend.equal(y_pred,label),'float32')

            return self.metric_per_label(true,pred)
        return dice
    


    def res_down(self,filters, downKernel,alpha_, input_):
        down_ = Conv2D(filters, downKernel, padding='same', 
                        kernel_initializer='he_uniform', 
                        kernel_regularizer=K.regularizers.l2(config1["L2PENALTY"]))(input_)
        down_ = LeakyReLU(alpha=alpha_)(down_)

        down_ = Conv2D(filters, downKernel, padding='same', 
                        kernel_initializer='he_uniform', 
                        kernel_regularizer=K.regularizers.l2(config1["L2PENALTY"]))(down_)
        down_res = LeakyReLU(alpha=alpha_)(down_)
        
        down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down_res)

        return down_pool, down_res


    def res_up(self,filters, upKernel,alpha_, input_, down_):
        upsample_ = UpSampling2D(size=(2, 2))(input_)

        up_ = Conv2D(filters, upKernel, padding='same', 
                    kernel_initializer='he_uniform', 
                    kernel_regularizer=K.regularizers.l2(config1["L2PENALTY"]))(upsample_)
        up_ = LeakyReLU(alpha=alpha_)(up_)
        up_ = Concatenate(axis=-1)([down_, up_])

        up_ = Conv2D(filters, upKernel, padding='same', 
                    kernel_initializer='he_uniform', 
                    kernel_regularizer=K.regularizers.l2(config1["L2PENALTY"]))(up_)
        up_ = LeakyReLU(alpha=alpha_)(up_)

        return up_

        
    def build(self,upKernel=3, downKernel=3, l2Penalty=config1["L2PENALTY"]):

        inputs = Input(shape=(config1["W"],config1["H"],config1["C"]))
        alpha_ = config1["LEAKYALPHA"]

        #first create encoder layers
        down0, down0_res = self.res_down(config1["FILTER1"], downKernel, alpha_ ,inputs)
        down1, down1_res = self.res_down(config1["FILTER2"], downKernel, alpha_, down0)
        down2, down2_res = self.res_down(config1["FILTER3"], downKernel, alpha_, down1)
        down3, down3_res = self.res_down(config1["FILTER4"], downKernel, alpha_, down2)
        down4, down4_res = self.res_down(config1["FILTER5"], downKernel, alpha_, down3)
        down5, down5_res = self.res_down(config1["FILTER6"], downKernel, alpha_, down4)

        print(down5.shape)
        #create bottleneck layer
        center = Conv2D(1024, (1, 1), padding='same',kernel_regularizer=K.regularizers.l2(l2Penalty))(down5)
        center = LeakyReLU(alpha=config1["LEAKYALPHA"])(center)
        print(center.shape)

        filterFrac = 16
        outputs = []
        #for each class create separate upsampling then concatenate them in the output
        for idx in range(self.numLabels):
            up5 = self.res_up(config1["FILTER6"]//filterFrac, upKernel, alpha_, center, down5_res)
            up4 = self.res_up(config1["FILTER5"]//filterFrac, upKernel, alpha_, up5, down4_res)
            up3 = self.res_up(config1["FILTER4"]//filterFrac, upKernel, alpha_, up4, down3_res)
            up2 = self.res_up(config1["FILTER3"]//filterFrac, upKernel, alpha_, up3, down2_res)
            up1 = self.res_up(config1["FILTER2"]//filterFrac, upKernel, alpha_, up2, down1_res)
            up0 = self.res_up(config1["FILTER1"]//filterFrac, upKernel, alpha_, up1, down0_res)

            #create sigmoid for each label
            #organMask = Conv2D(1, (1, 1), activation='sigmoid', name="label_"+str(idx), kernel_initializer='glorot_uniform')(up0)
            organMask = Conv2D(1, (1, 1), activation='relu', name="label_"+str(idx), kernel_initializer='glorot_uniform')(up0)


            #append it to outputs layer
            outputs.append(organMask)

        #concatenate all into single output
        labelSeg = Concatenate(axis=-1)(outputs)
        labelSeg = Activation('softmax',name="organ_output")(labelSeg)

        model = Model(inputs=inputs, outputs=[labelSeg],name="model")

        return model

    @staticmethod
    def weighted_batch_cross_entropy(y_true,y_pred):
        '''
        Calculates frequency of per batch, then weights cross-entropy accordingly
        '''
        
        # calculate frequency for each class
        freq = tf.reduce_sum(y_true,axis=(0,1,2)); #leave classes

        #calculate max freq
        max_freq = tf.reduce_max(freq)

        #normalize it by max
        freq_norm = max_freq/(freq+1); #avoid zero division

        # find classes that don't contribute
        w_mask = tf.equal(tf.cast(freq_norm,tf.int64),tf.cast(max_freq,tf.int64))
        weights = tf.where(w_mask,y=freq_norm,x=tf.ones_like(freq_norm))

        #take a square of weights to smooth
        weights = tf.pow(weights,0.3) #was working fine 0.2
        #print_weights = tf.Print(weights,[weights]) #print statement

        #calculate cross entropy
        y_pred /= tf.reduce_sum(y_pred,axis=len(y_pred.get_shape())-1,keep_dims=True)
        _epsilon = tf.convert_to_tensor(K.backend.epsilon(), dtype=y_pred.dtype.base_dtype)
        #clip bad values
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        
        # calculate weighted loss per class and batch
        weighted_losses = (y_true * tf.log(y_pred) + (1 - y_true) * tf.log(1 - y_pred))*weights
        
        return -tf.reduce_sum(weighted_losses,len(y_pred.get_shape()) - 1)

    @staticmethod
    def load_json_model(modelName):
        filePath = './checkpoint/'+modelName+".json";
        fileWeight = './checkpoint/'+modelName+"_weights.h5"

        with open(filePath,'r') as fp:
            json_data = fp.read();
        model = model_from_json(json_data)
        model.load_weights(fileWeight)

        return model

    def train_on_generator(self,trainGen,valGen):

        try:
            model = self.load_json_model(config1["modelName"])
            print("Loading model...");
        except Exception as e:
            print(e);
            print("Creating new model...")
            model = self.build()

        losses = {
            # "organ_output": "categorical_crossentropy"
            #"organ_output": weighted_cross_entropy
            #"organ_output": cross_entropy_multiclass
            "organ_output": self.weighted_batch_cross_entropy
            }
        lossWeights = {
            "organ_output": 1.0
            }

        print(model.summary())

        optimizer = K.optimizers.Adam(
            lr = config1["LEARNRATE"], decay = config3["DECAYRATE"]
        )

        esophagus_dice = self.get_label_dice(1)
        heart_dice = self.get_label_dice(2)
        trachea_dice = self.get_label_dice(3)
        aorta_dice = self.get_label_dice(4)

        #compile model
        model.compile(optimizer=optimizer,
                    loss = losses,#tot_loss,
                    loss_weights=lossWeights,
                    metrics=[esophagus_dice,heart_dice,trachea_dice,aorta_dice]
        )
                    

        #define callbacks
        modelCheckpoint = K.callbacks.ModelCheckpoint("./checkpoint/"+config1["modelName"]+"_weights.h5",
                                'val_loss',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='min', period=1)
        validationMetric = Metrics(valGen,config2["VALSTEPS"],config1["BATCHSIZE"])

        #save only model
        with open('./checkpoint/'+config1["modelName"]+'.json','w') as fp:
            fp.write(model.to_json())

        #fit model and store history
        hist = model.fit_generator(trainGen,
              steps_per_epoch = config2["STEPPEREPOCHS"],
              epochs = config1["NUMEPOCHS"],
              class_weight = 'auto',
              validation_data = valGen,
              validation_steps = config2["VALSTEPS"],
              verbose=0,
              callbacks=[modelCheckpoint,validationMetric])
        return hist

    
