#load current config file
from config import *

#add parent directory to path
import sys 
sys.path.append('..')
#load necessary modules
from modules import *

from modules.fileparser import FeatureLabelReader
from modules.processor import ImageProcessor
from modules.plotter import Plotter
from modules.datastats import save_train_stats


def lung_struct(roiName):
    #standarize name
    roiName = roiName.lower()
    #look for string matching
    if "esophagus" in roiName: return 1 
    elif "heart" in roiName: return 2 
    elif "spinalcord"==roiName: return 3
    elif "lung_l"==roiName: return 4
    elif "lung_r"==roiName: return 5  
    else: return 0; 

#compile SegTHOR dataset
def compile_dataset(pathToFiles,fileName):
    #test path in which feature label are same folder
    myParser = FeatureLabelReader(pathToFiles=pathToFiles)
    images = myParser.end_point_parser(filePath=pathToFiles,filePattern="Patient*.nii.gz")
    labels = myParser.end_point_parser(filePath=pathToFiles,filePattern="GT*.nii.gz")

    #now test compile dataset
    myParser.compile_dataset(imgFiles=images,labelFiles=labels,fileName=fileName,batchSize=1) #each nii contains ~220 slices

#compile LCTSC dataset
def compile_dataset2(pathToFiles,fileName):
    #test path in which feature label are same folder
    myParser = FeatureLabelReader(pathToFiles=pathToFiles,structNameMap=lung_struct)
    images = myParser.end_point_parser(filePath=pathToFiles,filePattern="*.dcm",dirPattern=["Scan","Train"])
    labels = myParser.end_point_parser(filePath=pathToFiles,filePattern="*.dcm",dirPattern=["RTst","Train"])

    #now test compile dataset
    myParser.compile_dataset(imgFiles=images,labelFiles=labels,fileName=fileName)


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
            k = plotter.plot_slice_label(imgBatch,labelBacth)
            # k = processor_unit_tester(imgBatch,processor=processor,plotter=plotter)

            if k==ord("d"): idx+=batchSize
            elif k==ord("a"): idx-=batchSize
            elif k==27: break   

if __name__=="__main__":

    #SegTHOR files
    #test path in which feature label are seperated
    pathToFiles = "../../SegmentationDataSets/SegTHOR/train/"
    fileName = "TRAIN_SegTHOR.h5"
    #pathToFiles = "../../SegmentationDataSets/SegTHOR/test/"
    fileName = "TEST_SegTHOR.h5"
    compile_dataset(pathToFiles,fileName)

    #LCTSC files
    #test path in which feature label are seperated
    pathToFiles = "../../SegmentationDataSets/LCTSC/"
    # fileName = "TRAIN_LCTSC.h5"
    # compile_dataset2(pathToFiles,fileName)


    #save_train_stats(fileName)
    test_random_images(fileName,batchSize=2)
    #save_label_idx_map(fileName)

