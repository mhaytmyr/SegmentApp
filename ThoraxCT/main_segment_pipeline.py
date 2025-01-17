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
from modules.model1 import SegTHOR

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
        labelBatch = label["organ_output"]
                
        print(imgBatch.shape, labelBatch.shape)
        #plot label and data
        k = plotter.plot_slice_label(imgBatch,labelBatch)
        if k==27: break

def label_merger(labelBatch, labelPred):
    '''
    Method to merge labels from two datasets. 
    In this case; LCTSC has esophagus, heart, L-R lung, spinal cord
                SegTHOR has esophagus, heart, aorta, trachea
    Algorithm: predict aorta and trachea from pretrained model and fuse it with LCTSC
    '''
    # print(np.unique(labelBatch))
    #create placeholder for fused labels
    labelFuse = np.zeros_like(labelBatch,dtype='uint8')
    #copy aorta and trachea from prediction
    labelFuse[labelPred.astype('uint8')==3] = 3
    labelFuse[labelPred.astype('uint8')==4] = 4

    #now take espophagus and heart ground truth
    labelFuse[labelBatch.astype('uint8')==1] = 1
    labelFuse[labelBatch.astype('uint8')==2] = 2
    
    return labelFuse

def fuse_labels(valGen,model,dataGenerator,outFile="TRAIN_MERGED.h5"):
    '''
    Method to create dataset that contains labels from both datasets (SegTHOR & LCTSC)
    '''

    #create instance of Plotter class
    plotter = Plotter()

    #create instance of FeatureLabelReader object, I will use "save_image_mask" method
    # reader = FeatureLabelReader()

    imgNum = 0
    # while imgNum<config1["VALSIZE"]:
    while imgNum<5981: #total number of slices in LCTSC file
        imgBatch, label = next(valGen)
        labelBatch = label["organ_output"]

        #denormalize image for plotting
        # imgBatchDeNorm = dataGenerator.denormalize(imgBatch)
        imgOriginal = dataGenerator.images

        #output of image is one-hot encoded, so take argmax
        #labelTrue = dataGenerator.morphological_operation(labelBatch.argmax(axis=-1).astype('uint8'),'open')
        labelOriginal = dataGenerator.labels

        #predict labels and apply inverse transformation
        labelPred = model.predict(imgBatch)
        labelPredUnZoom = dataGenerator.unzoom_label(labelPred,numClasses=5) #model predicts 5 labels, true labels contain 6
        labelPredDeCrop = dataGenerator.uncrop_label(labelPredUnZoom,numClasses=5)

        #get selected index of predictions
        labelPredMorph = dataGenerator.morphological_operation(labelPredDeCrop.argmax(axis=-1).astype('uint8'),'open')

        #create merged labels
        labelMerged = label_merger(labelOriginal,labelPredMorph)

        k = plotter.plot_label_prediction(imgOriginal,labelMerged,labelPredMorph)
        if k==27: break
        elif k==ord('s'):
            print("Saving ",imgNum)
            FeatureLabelReader.save_image_mask(imgOriginal,labelMerged,fileName=outFile)
        #print("Processed ",imgNum)
        imgNum += imgBatch.shape[0]

def merged_datasets(valGen,model,dataGenerator,outFile="TRAIN_MERGED.h5"):
    '''
    Method used to merge SegTHOR and LCTSC dataset
        SegTHOR has {'esophagus':1,'heart':2,'trachea':3,'aorta'}
        LCTSC has {'esophagus':1, 'heart':2, 'spinal cord':3 ,'L-R Lung':4,5}
    '''

    #create instance of Plotter class
    plotter = Plotter()
    imgNum = 0
    while imgNum<5981: #total number of slices in LCTSC file
        imgBatch, label = next(valGen)
        
        imgOriginal = dataGenerator.images
        labelOriginal = dataGenerator.labels
        
        #I am assigning different value to spinal cord
        #labelOriginal[labelOriginal==3] = 6
        labelOriginal[labelOriginal==4] = 0
        labelOriginal[labelOriginal==5] = 0
        labelOriginal[labelOriginal==3] = 0

        #look for slices that has non-zero label
        zSlices = np.array([np.max(labelOriginal[idx,...])>0 for idx in range(imgBatch.shape[0])],dtype=np.bool)

        print(imgOriginal.shape)
        
        imgSave = imgOriginal[zSlices,...]
        labelSave = labelOriginal[zSlices,...]
        print(imgSave.shape)
        if imgSave.shape[0]>0:
            print("Saving ",imgNum)
            FeatureLabelReader.save_image_mask(imgSave,labelSave,fileName=outFile)

        #k = plotter.plot_slice_label(imgOriginal,labelOriginal)
        #if k==27: break

        imgNum += imgBatch.shape[0]

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


import pickle
if __name__=='__main__':
    
    arg = sys.argv[1]
    gpu = sys.argv[2]

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu)

   
    #define train file name
    trainFile = "TRAIN_SegTHOR.h5"
    # trainFile = "TRAIN_MERGED.h5"
    # testFile = "TRAIN_LCTSC.h5"
    testFile = "TEST_SegTHOR.h5"

    #now get normalization parameters
    imgMean, imgStd = get_normalization_param(trainFile.replace(".h5","_STAT.h5"))

    #create data generator
    dataGenerator = DataGenerator(normalize={"means":imgMean,"vars":imgStd})

    #create train data generator
    trainGen = dataGenerator.generate_data(trainFile,
                            # batchSize=2,
                            batchSize=config1["BATCHSIZE"],
                            augment=True,shuffle=True)
    #create test data generator
    valGen = dataGenerator.generate_data(testFile,
                           batchSize=2,
                        #    batchSize=config1["BATCHSIZE"],
                           augment=False,shuffle=False)
     

    arg = sys.argv[1]
    if arg=='train':
        myModel = SegTHOR(numLabels=config1["NUMCLASSES"])
        hist = myModel.train_on_generator(trainGen,valGen)

        with open('./log/'+config1["modelName"]+".log", 'wb') as fp:
            pickle.dump(hist.history, fp)
        history = hist.history

        fig, axes = plt.subplots(ncols=2,nrows=1)
        ax = axes.ravel();
        ax[0].plot(history['loss'],'r*',history['val_loss'],'g^')
        ax[0].legend(labels=["Train loss","Val Loss"],loc="best")
        ax[0].set_title("Cross entropy loss vs epochs")        
        ax[0].set_xlabel("# of epochs")
        ax[1].plot(history["dice_1"],"r*",history["dice_2"],"g^",history["dice_3"],"b>",history["dice_4"],"k<")
        ax[1].legend(labels=["Esophagus","Heart","Trachea","aorta","Spinal Cord"])
        ax[1].set_title("Dice coefficients vs epochs")
        ax[1].set_xlabel("# of epochs");
        fig.savefig("./log/"+config1["modelName"]+".jpg")

        
    elif arg=='test':
        model = SegTHOR.load_json_model(config1["modelName"])
        plot_prediction(valGen,model,dataGenerator)
        # plot_prediction(trainGen,model,dataGenerator)

    elif arg=='submit':
        #load model
        model = SegTHOR.load_json_model(config1["modelName"])
        #creat instance of SubmitPrediction
        predictor = SubmitPrediction(
                            pathToImages='../../SegmentationDataSets/SegTHOR/',
                            filePattern='Patient'
                            )
        predictor.set_model(model)
        predictor.set_normalization(normParam={"means":imgMean,"vars":imgStd})
        predictor.predict_nii_patients(batchSize=8)      

    elif arg=='plot':
        test_generator(valGen,dataGenerator)
        # test_generator(trainGen,dataGenerator)

    elif arg=='fuse':
        model = SegTHOR.load_json_model(config1["modelName"])
        merged_datasets(valGen,model,dataGenerator,outFile="TRAIN_MERGED.h5")
        #fuse_labels(valGen,model,dataGenerator)
