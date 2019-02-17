from modules.fileparser import FeatureLabelReader

def prostate_struct(roiName):
    #standarize name
    roiName = roiName.lower()
    #look for string matching
    if "body" in roiName: return 1 
    elif "bexiga" in roiName or "bladder" in roiName: return 2 
    elif "reto"==roiName or "rectum"==roiName: return 3 
    elif "cfd" in roiName or "cfe" in roiName or "femur" in roiName: return 4
    elif "ctv" in roiName: return 5
    elif "sigmoide"==roiName: return 6 
    else: return 0; 

def sbrt_lung_struct(roiName):
    #standarize name
    roiName = roiName.lower()
    #look for string matching
    if "body" in roiName: return 1 
    elif "gtv_research_lun" in roiName: return 2 
    elif "gtv_research_med"==roiName: return 3  
    else: return 0; 

def head_neck_struct(roiName):
    #standarize name
    roiName = roiName.lower()
    #look for string matching
    if "body" in roiName: return 1 
    elif "chiasm" in roiName: return 2 
    elif "brainstem" in roiName and "+" not in roiName and "_" not in roiName: return 3 
    elif "cord" in roiName and "+" not in roiName and "_" not in roiName: return 4
    elif "parotid" in roiName and "ptv" not in roiName: return 5
    elif "mandible" in roiName: return 6 
    elif "on_l" in roiName or "on_r" in roiName or "optic n" in roiName: return 7 #optical nerve LR
    else: return 0; 

def end_point_unit_tester():

    #test path in which feature label are seperates
    pathToFiles = "../../SegmentationDataSets/CHAOS/CT_data_batch1/"
    myParser = FeatureLabelReader(pathToFiles=pathToFiles)
    images = myParser.end_point_parser(filePattern="i*.dcm")
    labels = myParser.end_point_parser(filePattern="liver_GT*.png")

    #length has to be equal
    assert(images.__len__()==labels.__len__())

    #test path in which feature label are same folder
    pathToFiles = "../../SegmentationDataSets/SegTHOR/train/"
    images = myParser.end_point_parser(filePath=pathToFiles,filePattern="Patient*.nii.gz")
    labels = myParser.end_point_parser(filePath=pathToFiles,filePattern="GT*.nii.gz")

    #length has to be equal
    assert(images.__len__()==labels.__len__())


    #test path in which CT and RS in different directory 
    pathToFiles = "../TrainData/"
    images = myParser.end_point_parser(filePath=pathToFiles,dirPattern=["PROSTATE","CT"],filePattern="*.dcm")
    labels = myParser.end_point_parser(filePath=pathToFiles,dirPattern=["PROSTATE","RTst"],filePattern="*.dcm")


    #test path in which CT and RS struct in same directory 
    pathToFiles = "../../SegmentationDataSets/SBRT_Lung/"
    images = myParser.end_point_parser(filePath=pathToFiles,filePattern="CT*.dcm")
    labels = myParser.end_point_parser(filePath=pathToFiles,filePattern="RS*.dcm")

    #test only images
    pathToFiles = "../ValData/"
    images = myParser.end_point_parser(filePath=pathToFiles)
    print(images)


def compile_dataset_unit_tester():

    #test path in which feature label are seperated
    pathToFiles = "../../SegmentationDataSets/CHAOS/CT_data_batch1/"
    myParser = FeatureLabelReader(pathToFiles=pathToFiles,structNameMap=prostate_struct)
    images = myParser.end_point_parser(filePattern="i*.dcm")
    labels = myParser.end_point_parser(filePattern="liver_GT*.png")

    #length has to be equal
    assert(images.__len__()==labels.__len__())

    #testing RS file format 
    pathToFiles = "../../SegmentationDataSets/SBRT_Lung/"
    myParser = FeatureLabelReader(pathToFiles=pathToFiles,structNameMap=sbrt_lung_struct)
    images = myParser.end_point_parser(filePath=pathToFiles,filePattern="CT*.dcm")
    labels = myParser.end_point_parser(filePath=pathToFiles,filePattern="RS*.dcm")

    #now test compile dataset
    myParser.compile_dataset(imgFiles=images,labelFiles=labels,fileName="TRAIN_LUNG.h5")

    #test path in which CT and RS in different directory 
    pathToFiles = "../TrainData/"
    images = myParser.end_point_parser(filePath=pathToFiles,dirPattern=["PROSTATE","CT"],filePattern="*.dcm")
    labels = myParser.end_point_parser(filePath=pathToFiles,dirPattern=["PROSTATE","RTst"],filePattern="*.dcm")

    #now test compile dataset
    #myParser.compile_dataset(imgFiles=images,labelFiles=labels,fileName="TRAIN_PROSTATE.h5")

from modules.plotter import Plotter
from modules.processor import ImageProcessor
import h5py

def test_random_images(fileName,batchSize=6):

    #impor Image Processor
    processor = ImageProcessor()

    #create instance of Plotter class
    plotter = Plotter()
    
    with h5py.File(fileName,"r") as organFile:
        data = organFile["features"]
        labels = organFile["labels"]

        idx, n = 0, labels.shape[0]

        while idx<n:
            imgBatch = data[idx:idx+batchSize]
            labelBacth = labels[idx:idx+batchSize]

            #plot label and data
            #k = plotter.plot_slice_label(imgBatch,labelBacth)
            k = processor_unit_tester(imgBatch,processor=processor,plotter=plotter)

            if k==ord("d"): idx+=batchSize
            elif k==ord("a"): idx-=batchSize
            elif k==27: break   

def processor_unit_tester(slices,processor=None,plotter=None):
        #apply preprocessing
        imgStandard = processor.standardize_img(slices)
        imgCrop = processor.crop(imgStandard)
        imgZoom = processor.zoom(imgCrop)
        #imgNorm = processor.normalize(imgZoom)
        
        #apply inverse preprocessing
        #imgDeNorm = processor.denormalize(imgNorm)
        #imgUnZoom = processor.unzoom(imgDeNorm)
        imgUnZoom = processor.unzoom(imgZoom)
        imgDeCrop = processor.uncrop(imgUnZoom)
        
        
        #imgZoomUnZoom = [imgCrop[idx],imgUnZoom[idx]]
        #print(abs(imgCrop[idx]-imgUnZoom[idx]).max())
        #imgCropUnCrop = [imgStandard[idx],imgDeCrop[idx],imgStandard[idx]-imgDeCrop[idx]]
        #imgStandUnStand = [imgStandard[idx],imgDeStandard[...,idx]]
        
        #labelPred = self.model.predict(processor.img_to_tensor(imgBatch))
        #labelPredUnZoom = processor.unzoom_label(labelPred)
        #labelPredDeCrop = processor.uncrop_label(labelPredUnZoom)
        #labelPredDeStandard = processor.de_standardize_nii(labelPredDeCrop.argmax(axis=-1))
        
        #k = plotter.plot_slice_label(imgDeNorm[idx:idx+batchSize],labelPred)
        k = plotter.plot_slices([imgStandard[0],imgDeCrop[0]])
        #k = plotter.plot_slices([imgDeStandard,processor.images])
        return k

if __name__=='__main__':
    #end_point_unit_tester()
    #compile_dataset_unit_tester()
    #test_random_images("TRAIN_LIVER.h5")
    test_random_images("TRAIN_PROSTATE.h5")
    #test_random_images("TRAIN_LUNG.h5")