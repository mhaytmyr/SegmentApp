#package specific imports
from modules import *

#module specific imports
import keras as K
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


class SegTHOR:
    def __init__(self,numLabels):
        #number of output labels
        self.numLabels = numLabels

    def res_down(self,filters, downKernel,alpha_, input_):
        down_ = Conv2D(filters, downKernel, padding='same', kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(input_)
        down_ = LeakyReLU(alpha=alpha_)(down_)

        down_ = Conv2D(filters, downKernel, padding='same', kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(down_)
        down_res = LeakyReLU(alpha=alpha_)(down_)
        
        down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down_res)

        return down_pool, down_res


    def res_up(self,filters, upKernel,alpha_, input_, down_):
        upsample_ = UpSampling2D(size=(2, 2))(input_)

        up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(upsample_)
        up_ = LeakyReLU(alpha=alpha_)(up_)
        up_ = Concatenate(axis=-1)([down_, up_])

        up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='he_uniform', kernel_regularizer=K.regularizers.l2(L2PENALTY))(up_)
        up_ = LeakyReLU(alpha=alpha_)(up_)

        return up_

        
    def build(self,upKernel=3, downKernel=3, l2Penalty=config1["L2PENALTY"]):

        inputs = Input(shape=(config1["W"],config1["H"],config1["C"]))
        alpha_ = config1["LEAKYALHPA"]

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

        filterFrac = 8
        outputs = []
        #for each class create separate upsampling then concatenate them in the output
        for idx in self.numLabels:
            up5 = self.res_up(config1["FILTER6"]//filterFrac, upKernel, alpha_, center, down5_res)
            up4 = self.res_up(config1["FILTER5"]//filterFrac, upKernel, alpha_, up5, down4_res)
            up3 = self.res_up(config1["FILTER4"]//filterFrac, upKernel, alpha_, up4, down3_res)
            up2 = self.res_up(config1["FILTER3"]//filterFrac, upKernel, alpha_, up3, down2_res)
            up1 = self.res_up(config1["FILTER2"]//filterFrac, upKernel, alpha_, up2, down1_res)
            up0 = self.res_up(config1["FILTER1"]//filterFrac, upKernel, alpha_, up1, down0_res)

            #create sigmoid for each label
            organMask = Conv2D(1, (upKernel, upKernel), activation='sigmoid', name="label_"+str(idx), kernel_initializer='glorot_uniform')(up0)

            #append it to outputs layer
            outputs.append(organMask)

        #concatenate all into single output
        #labelSeg = Conv2D(config1["NUMCLASSES"], (1, 1), activation='softmax', name='organ_output', kernel_initializer='glorot_uniform')(outputs)
        labelSeg = Concatenate(axis=-1)(outputs)
        labelSeg = Activation('softmax',name="organ_output")(labelSeg)

        model = Model(inputs=inputs, outputs=[labelSeg],name="model");

        return model

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

        model.summary()
        

        