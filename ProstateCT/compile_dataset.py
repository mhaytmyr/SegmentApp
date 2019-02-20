from config import *

#add parent directory to path
import sys 
sys.path.append('..')
#load necessary modules
from modules import *

pdb.set_trace()


from modules.fileparser import FeatureLabelReader
from modules.fileparser_unit_test import test_random_images
from modules.datastats import save_train_stats
pdb.set_trace()

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

def compile_dataset(pathToFiles,fileName):
    myParser = FeatureLabelReader(pathToFiles=pathToFiles,structNameMap=prostate_struct)
    images = myParser.end_point_parser(dirPattern=["CT"],filePattern="*.dcm")
    labels = myParser.end_point_parser(dirPattern=["RTst"],filePattern="*.dcm")

    #now test compile dataset
    myParser.compile_dataset(imgFiles=images,labelFiles=labels,fileName=fileName)

if __name__=="__main__":

    #test path in which feature label are seperated
    # pathToFiles = "../../SegmentationDataSets/RCJr_Prostate/Train/"
    fileName = "TRAIN_PROSTATE.h5"
    pathToFiles = "../../SegmentationDataSets/RCJr_Prostate/Test/"
    #fileName = "TEST_PROSTATE.h5"
    # compile_dataset(pathToFiles,fileName)

    #save_train_stats(fileName)
    #test_random_images(fileName,batchSize=2)
    #save_label_idx_map(fileName)

