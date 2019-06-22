#package specific imports
from modules import *

#module specific imports
import keras as K
import tensorflow as tf
from keras.engine.topology import get_source_inputs
from keras.applications import imagenet_utils
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.utils import generic_utils

from keras.backend import tf as ktf
from keras.layers import SpatialDropout2D, Activation, Flatten, Reshape, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Layer, MaxPooling2D, UpSampling2D, Conv2D, GlobalMaxPooling2D
from keras.layers import Lambda, Input, BatchNormalization, Concatenate, Add, Multiply, Dense
from keras.models import Model, model_from_json
from keras.layers.advanced_activations import PReLU, LeakyReLU

from modules.custom_layers import Metrics


class DCGAN:
    def __init__(self,numLabels):
        # number of output labels
        self.numLabels = numLabels
        self.label_models = []
        
        # create optimizer
        optimizer = K.optimizers.SGD(lr = config1["LEARNRATE"])
        
        # Build and compile the discriminator
        #self.discriminator = self.load_json_model('Discriminator_90')
        #self.discriminator = self.load_json_model(config1["modelName"])
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss={'out':'binary_crossentropy'}, 
               optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        #self.generator = self.load_json_model('Generator_90')
        print(self.generator.summary())

        # The generator takes noise as input and generates imgs
        z = Input(shape=(config1['LATENTDIM'],))
        img = self.generator(z)

        # The discriminator takes generated images as input and determines validity
        valid, *tmp = self.discriminator(img)

        # create optimizer
        optimizer = K.optimizers.Adam(lr = config1["LEARNRATE"])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='mse', optimizer=optimizer)

    def res_down(self,filters, downKernel, alpha_, input_, name_ = None):
        down_ = Conv2D(filters, downKernel, padding='same', 
                        kernel_initializer='he_uniform', 
                        kernel_regularizer=K.regularizers.l2(config1["L2PENALTY"]),
                        bias_initializer='zeros', 
                        bias_regularizer=K.regularizers.l2(config1["L2PENALTY"])
                        )(input_)
        down_ = BatchNormalization()(down_)
        down_ = LeakyReLU(alpha=alpha_)(down_)

        down_ = Conv2D(filters, downKernel, padding='same', 
                        kernel_initializer='he_uniform', 
                        kernel_regularizer=K.regularizers.l2(config1["L2PENALTY"]),
                        bias_initializer='zeros', 
                        bias_regularizer=K.regularizers.l2(config1["L2PENALTY"])
                        )(down_)
        down_res = LeakyReLU(alpha=alpha_, name=name_)(down_)
        down_pool = AveragePooling2D((2, 2), strides=(2, 2))(down_res)

        return down_pool, down_res

    def generator_up(self, filters, upKernel, alpha_, input_, name_=None):
        '''
        Convolution layers to upsample
        '''
        upsample_ = UpSampling2D(size=(2, 2))(input_)

        up_ = Conv2D(filters, upKernel, padding='same', 
                        kernel_initializer='he_uniform', 
                        kernel_regularizer=K.regularizers.l2(config1["L2PENALTY"]),
                        bias_initializer='zeros', 
                        bias_regularizer=K.regularizers.l2(config1["L2PENALTY"])
                        )(upsample_)
        up_ = BatchNormalization()(up_)
        up_ = LeakyReLU(alpha=alpha_)(up_)

        return up_
 
    def build_generator(self, upKernel=3, l2Penalty=config1["L2PENALTY"], latent_dim=config1["LATENTDIM"]):
        '''
        Generator method that creates realistic CT image from noise
        '''
        alpha_ = config1["LEAKYALPHA"]

        #latent layer that generates 4x6x512 latent image image from 100 noise
        input_ = Input(shape=(latent_dim,),name='gen_input')
        z = Dense(4*6*config1["FILTER4"], activation='linear', 
            kernel_initializer='he_uniform', 
            kernel_regularizer=K.regularizers.l2(config1["L2PENALTY"]),
            bias_initializer='zeros', 
            bias_regularizer=K.regularizers.l2(config1["L2PENALTY"])
            )(input_)
        z = Reshape((4,6,config1["FILTER4"]))(z)

        up5 = self.generator_up(config1["FILTER4"], upKernel, alpha_, z, 'gen_up5')
        up4 = self.generator_up(config1["FILTER3"], upKernel, alpha_, up5, 'gen_up4')
        up3 = self.generator_up(config1["FILTER2"], upKernel, alpha_, up4, 'gen_up3')
        up2 = self.generator_up(config1["FILTER2"], upKernel, alpha_, up3, 'gen_up2')
        up1 = self.generator_up(config1["FILTER1"], upKernel, alpha_, up2, 'gen_up1')
        up0 = self.generator_up(config1["FILTER1"], upKernel, alpha_, up1, 'gen_up0')

        #final layer
        output = Conv2D(1, (1, 1), padding='same', name='gen_out',
                kernel_initializer='he_uniform', 
                kernel_regularizer=K.regularizers.l2(config1["L2PENALTY"]),
                bias_initializer='zeros',
                bias_regularizer=K.regularizers.l2(config1["L2PENALTY"]),
                )(up0)

        #create model and return
        model = Model(inputs=input_, name="generator", outputs=[output])

        return model

    def build_discriminator(self, downKernel=3, l2Penalty=config1["L2PENALTY"]):
        '''
        Dicriminator method that identifies fake image from real image
        '''
        inputs_ = Input(shape=(config1["W"],config1["H"],config1["C"]))
        alpha_ = config1["LEAKYALPHA"]

        #first create encoder layers
        down0, down0_res = self.res_down(config1["FILTER1"]//2, downKernel, alpha_ ,inputs_, 'down0_res')
        down1, down1_res = self.res_down(config1["FILTER2"]//2, downKernel, alpha_, down0, 'down1_res')
        down2, down2_res = self.res_down(config1["FILTER3"]//2, downKernel, alpha_, down1, 'down2_res')
        down3, down3_res = self.res_down(config1["FILTER4"]//2, downKernel, alpha_, down2, 'down3_res')
        down4, down4_res = self.res_down(config1["FILTER5"]//2, downKernel, alpha_, down3, 'down4_res')
        down5, down5_res = self.res_down(config1["FILTER6"]//2, downKernel, alpha_, down4, 'down5_res')

        #create bottleneck layer
        center = Conv2D(1024, (1, 1), padding='same', kernel_regularizer=K.regularizers.l2(l2Penalty))(down5)
        center = LeakyReLU(alpha=config1["LEAKYALPHA"],name="bottle_neck")(center)
        #create global averaging layer
        output_ = GlobalAveragePooling2D()(center)
        #create output classification layer
        output_ = Dense(1, activation='sigmoid',name="out")(output_)
        #create model
        model = Model(inputs=inputs_, name="dicriminator",
                    outputs=[output_, down0_res, down1_res, down2_res, down3_res, down4_res, down5_res])

        return model

    def train_generator(self, trainGen, valGen, compile=True):
        '''
        Generates images from random noise, then uses pretrained discriminator to train generator
        '''

        #start training
        for epoch in range(config1["NUMEPOCHS"]):
            for step in range(config2["STEPPEREPOCHS"]):
                #get data
                (real_imgs, _) = next(trainGen)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random half of images
                #idx = np.random.randint(0, features.shape[0], config1["BATCHSIZE"])
                #real_img = features[idx]

                # Sample noise and generate a batch of new images
                noise = np.random.normal(0, 1, (real_imgs.shape[0], config1["LATENTDIM"]))
                fake_imgs = self.generator.predict(noise)

                #combine images
                #X_in = np.concatenate([real_imgs, fake_imgs])
                #create soft labels
                valid = np.random.uniform(low=0.8,high=1.0,size=(real_imgs.shape[0], 1))
                fake = np.random.uniform(low=0.0,high=.3,size=(real_imgs.shape[0], 1))
                #valid = np.ones((real_imgs.shape[0], 1))
                #fake = np.zeros((real_imgs.shape[0], 1))
                #y_out = np.concatenate([valid,fake])

                #shuffle data
                #X_in,y_out = self.unison_shuffled_copies(X_in,y_out)

                # For the combined model we will only train the generator
                self.discriminator.trainable = True

                # Train the discriminator (fake classified as ones and generated as zeros)
                #d_loss = self.discriminator.train_on_batch(X_in, {"out":y_out})
                d_loss_real = self.discriminator.train_on_batch(real_imgs, {"out":valid})
                d_loss_fake = self.discriminator.train_on_batch(fake_imgs, {"out":fake})
                #d_loss = self.discriminator.evaluate(X_in, {"out":y_out},verbose=False)
                #d_loss_real = self.discriminator.evaluate(real_imgs, {"out":valid}, verbose=False)
                #d_loss_fake = self.discriminator.evaluate(fake_imgs, {"out":fake}, verbose=False)
                # metrics: ['loss', 'out_loss', 'out_acc']
                d_loss = 0.5*np.add(d_loss_fake,d_loss_real)

                # For the combined model we will only train the generator
                self.discriminator.trainable = False

                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator (wants discriminator to mistake images as real)
                if np.random.choice([1,1,1,0])==1:
                    valid = np.ones((noise.shape[0], 1))
                else:
                    valid = np.zeros((noise.shape[0], 1))

                g_loss = self.combined.train_on_batch(noise, valid)
                # Plot the progress
                if step % 10 == 0:
                    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[1], 100*d_loss[2], g_loss))

                # If at save interval => save generated image samples
                #if epoch % save_interval == 0:
                #    self.save_imgs(epoch)
            
            # save model after each epoch
            # TODO: need to add callbacks
            if epoch % 10 == 0:
                print("Saving models ...")
                self.save_json_model(self.discriminator, f"Discriminator_{epoch}")
                self.save_json_model(self.generator, f"Generator_{epoch}")

    def train_discriminator(self,trainGen,valGen,compile=True):
        '''
        Trains dicriminator to identify fake image from real image
        Here: trainGen and valGen will input images that are synthetically generated
        '''
        try:
            model = self.load_json_model(config1["modelName"])
            print("Loading model...");
        except Exception as e:
            print(e);
            print("Creating new model...")
            model = self.build_discriminator()

        losses = {"out": "binary_crossentropy"}
        lossWeights = { "out": 1.0}

        #compile model with different lr and 
        if compile:
            optimizer = K.optimizers.Adam(lr = config1["LEARNRATE"])

            #compile model if it has not been loaded before
            model.compile(optimizer=optimizer,
                    loss = losses, 
                    metrics= [K.metrics.binary_accuracy]
            )

        print(model.summary())

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
              verbose=1,
              callbacks=[modelCheckpoint,validationMetric])

        return hist

    

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

        #create bottleneck layer
        center = Conv2D(1024, (1, 1), padding='same',kernel_regularizer=K.regularizers.l2(l2Penalty))(down5)
        center = LeakyReLU(alpha=config1["LEAKYALPHA"])(center)
        print(center.shape)


        filterFrac = 4
        outputs = []
        #for each class create separate upsampling then concatenate them in the output
        for idx in range(self.numLabels):
            up5 = self.res_up(config1["FILTER6"]//filterFrac, upKernel, alpha_, center, down5_res )
            up4 = self.res_up(config1["FILTER5"]//filterFrac, upKernel, alpha_, up5, down4_res)
            up3 = self.res_up(config1["FILTER4"]//filterFrac, upKernel, alpha_, up4, down3_res)
            up2 = self.res_up(config1["FILTER3"]//filterFrac, upKernel, alpha_, up3, down2_res)
            up1 = self.res_up(config1["FILTER2"]//filterFrac, upKernel, alpha_, up2, down1_res)
            up0 = self.res_up(config1["FILTER1"]//filterFrac, upKernel, alpha_, up1, down0_res)

            #create sigmoid for each label
            #organMask = Conv2D(1, (1, 1), activation='sigmoid', name="label_"+str(idx), kernel_initializer='glorot_uniform')(up0)
            organMask = Conv2D(1, (1, 1), activation='relu', kernel_initializer='glorot_uniform')(up0)
            organMaskOut = Activation('sigmoid', name="label_"+str(idx))(organMask)

            #create separate model for each output
            curr_model = Model(inputs=inputs, outputs=[organMaskOut])
            curr_model.compile(loss="binary_crossentropy", optimizer=self.class_optimizer,metrics=["acc"])
            self.label_models.append(curr_model)

            #append it to outputs layer
            outputs.append(organMask)

        #create reconstruction layer first
        recon = self.recon_up(center,alpha_)
        print(recon.shape)
        curr_model = Model(inputs=inputs, outputs=[recon])
        curr_model.compile(loss="mean_absolute_error", optimizer=self.class_optimizer, metrics=["mse"])
        self.label_models.append(curr_model)
   

        #concatenate all into single output to create combined model
        labelSeg = Concatenate(axis=-1)(outputs)
        labelSeg = Activation('softmax',name="organ_output")(labelSeg)
        self.main_model = Model(inputs=inputs, outputs=[labelSeg],name="model")

        return self.main_model

    @staticmethod
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    @staticmethod
    def save_json_model(model, modelName):
        filePath = './checkpoint/'+modelName+".json";
        fileWeight = './checkpoint/'+modelName+"_weights.h5"

        model_json = model.to_json()
        with open(filePath, "w") as json_file:
            json_file.write(model_json)
        
        # serialize weights to HDF5
        model.save_weights(fileWeight)
        return 0

    @staticmethod
    def load_json_model(modelName):
        filePath = './checkpoint/'+modelName+".json";
        fileWeight = './checkpoint/'+modelName+"_weights.h5"

        with open(filePath,'r') as fp:
            json_data = fp.read();
        model = model_from_json(json_data)
        model.load_weights(fileWeight)

        return model

    
