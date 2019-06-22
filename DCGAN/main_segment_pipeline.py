#load current config file
from config import *

#add parent directory to path
import sys 
sys.path.append('..')
#load necessary modules
from modules import *


#module specific imports
from keras.models import model_from_json
from modules.data_generator import DataGenerator
from modules.plotter import Plotter
from modules.fileparser import FeatureLabelReader
from modules.model2 import DCGAN

def test_generator(myGen,dataGenerator):

    #create instance of Plotter class
    plotter = Plotter()

    while True:
        #images are already processed in 
        imgBatch, label = next(myGen)

        #denormalize image for plotting
        imgBatch = dataGenerator.denormalize(imgBatch)

        #print augmentation
        if len(dataGenerator.augments)>0:
            print(dataGenerator.augments)

        #gen output is dictionary
        labelBatch = label["out"]
                
        print(imgBatch.shape, labelBatch.shape)
        #plot label and data
        k = plotter.plot_slice_label(imgBatch,labelBatch)
        if k==27: break

def test_dicriminator(myGen,dataGenerator):

    #create instance of Plotter class
    plotter = Plotter()

    while True:
        #images are already processed in 
        imgBatch, label = next(myGen)

        #denormalize image for plotting
        imgBatch = dataGenerator.denormalize(imgBatch)

        #gen output is dictionary
        labelBatch = label["out"]
                
        print(imgBatch.shape, labelBatch)
        #plot label and data
        k = plotter.plot_slices(imgBatch)
        if k==27: break

from scipy.misc import imsave
def plot_prediction(valGen,model,dataGenerator):
 
    #create instance of Plotter class
    plotter = Plotter()

    imgNum = 0
    while imgNum<config1["VALSIZE"]:
        imgBatch, label = next(valGen)
        labelBatch = label["organ_output"]

        #denormalize image for plotting
        imgBatchDeNorm = dataGenerator.denormalize(imgBatch)       

        #output of image is one-hot encoded, so take argmax
        #labelTrue = dataGenerator.morphological_operation(labelBatch.argmax(axis=-1).astype('uint8'),'open')
        labelTrue = labelBatch.argmax(axis=-1)

        #predict image and normalize first
        labelPred = model.predict(imgBatch)

        #get selected index of predictions
        labelPred = dataGenerator.morphological_operation(labelPred.argmax(axis=-1).astype('uint8'),'open')

        k = plotter.plot_label_prediction(imgBatchDeNorm,labelTrue,labelPred)
        if k==27: break
        if k==ord('s'):
            imsave("true.png",labelTrue[0].astype('uint8'))
            imsave("pred.png",labelPred[0].astype('uint8'))
        imgNum += imgBatch.shape[0]
        
        #ax = overlay_contours_plot(pred_organ,true_organ,image,imgNum,ax); 
        #overlay_contours_save(pred_organ,true_organ,image,imgNum);         

def get_normalization_param(trainFile):
    with h5py.File(trainFile,"r") as fp:
        means = fp["means"][0];
        variance = fp["variance"][0]
    return means, variance 


def plot_losses(modelName):

    try:
        val = int(gpu)
        print("Input integer")
        with open('./log/'+modelName+".log", 'rb') as fp:
            data = pickle.load(fp)
    except ValueError:
        print("Input string")
        with open(modelName, 'rb') as fp:
            data = pickle.load(fp)    

    fig, axes = plt.subplots(ncols=3,nrows=2,figsize=(15,8))
    ax = axes.ravel()
    ax[0].plot(data['loss'],'r*',data['val_loss'],'g^');
    ax[0].legend(labels=["Train loss","Val Loss"],loc="best");
    ax[0].set_title("Cross entropy loss vs epochs")        
    #ax[0].set_xlabel("# of epochs");

    # ax[1].plot(data["lr"],"b>")
    # ax[1].set_title("Learning Rate")
    ax[1].plot(np.array(data['val_loss'])-np.array(data['loss']),"b>")
    ax[1].set_title("Generilzation Error")


    ax[2].plot(data["dice"],"r*",data["val_dice"],"g^")
    ax[2].legend(labels=["Train","Val"])
    ax[2].set_title("Esophagus Dice coefficients")
    #ax[1].set_xlabel("# of epochs")

    ax[3].plot(data["dice_1"],"r*",data["val_dice_1"],"g^")
    ax[3].legend(labels=["Train","Val"])
    ax[3].set_title("Heart Dice coefficients")
    ax[3].set_xlabel("# of epochs")

    ax[4].plot(data["dice_2"],"r*",data["val_dice_2"],"g^")
    ax[4].legend(labels=["Train","Val"])
    ax[4].set_title("Trachea Dice coefficients")
    ax[4].set_xlabel("# of epochs")
    
    ax[5].plot(data["dice_3"],"r*",data["val_dice_3"],"g^")
    ax[5].legend(labels=["Train","Val"])
    ax[5].set_title("Aorta Dice coefficients")
    ax[5].set_xlabel("# of epochs")

    #fig.savefig("./log/"+modelName+".jpg")

import pickle
if __name__=='__main__':
    
    arg = sys.argv[1]
    gpu = sys.argv[2]

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu)
    
    #cropped images
    trainFile = "../../Segmentation/HeartAortaTracheaEsophagus/TRAIN_SegTHOR_CROP.h5"
    testFile = "../../Segmentation/HeartAortaTracheaEsophagus/TEST_SegTHOR_CROP.h5"

    #now get normalization parameters
    imgMean, imgStd = get_normalization_param(trainFile.replace(".h5","_STAT.h5"))

    #create data generator
    dataGenerator = DataGenerator(normalize={"means":imgMean,"vars":imgStd},crop=False)

    #create train data generator
    trainGen = dataGenerator.data_generator_discriminator(trainFile,
                            shuffle=False,
                            batchSize=config1["BATCHSIZE"], create_fake=False
                            )
    #create test data generator
    valGen = dataGenerator.data_generator_discriminator(testFile,
                           shuffle=False,
                           batchSize=config1["BATCHSIZE"], create_fake=False
                           )
     

    arg = sys.argv[1]
    if arg=='train':
        myModel = DCGAN(numLabels=config1["NUMCLASSES"])
        myModel.train_discriminator(trainGen, valGen)

    elif arg=='train_gen':
        myModel = DCGAN(numLabels=config1["NUMCLASSES"])
        myModel.train_generator(trainGen, valGen)

    elif arg=='loss':
        plot_losses(gpu)
            
    elif arg=='test':
        generator = DCGAN.load_json_model('Generator_290')
        plotter = Plotter()

        while True:
            noise = np.random.normal(0, 1, (4, config1["LATENTDIM"]))
            generated = generator.predict(noise)
            
            imgNorm = np.zeros_like(generated)
            for i in range(imgNorm.shape[0]):
                imgNorm[i] = (generated[i]-generated[i].min()) / (generated[i].max() - generated[i].min())

            k = plotter.plot_slices(imgNorm)
            if k==27: break
        # plot_prediction(trainGen,model,dataGenerator)

    elif arg=='plot':
        #test_dicriminator(valGen,dataGenerator)
        test_dicriminator(trainGen,dataGenerator)
