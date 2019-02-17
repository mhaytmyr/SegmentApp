#add parent directory to path
import sys
sys.path.append('..')
#load necessary modules
from modules.fileparser import FeatureLabelReader
from modules.fileparser_unit_test import test_random_images

def compile_dataset(pathToFiles,fileName):
    myParser = FeatureLabelReader(pathToFiles=pathToFiles)
    images = myParser.end_point_parser(filePattern="*.dcm",dirPattern=["DICOM_anon"])
    labels = myParser.end_point_parser(filePattern="liver_GT*.png",dirPattern=["Ground"])

    #length has to be equal
    assert(images.__len__()==labels.__len__())

    #now test compile dataset
    myParser.compile_dataset(imgFiles=images,labelFiles=labels,fileName=fileName)

if __name__=="__main__":

    #test path in which feature label are seperated
    pathToFiles = "../../SegmentationDataSets/CHAOS/"
    fileName = "TRAIN_LIVER.h5"
    # compile_dataset(pathToFiles,fileName)

    #save_train_stats(fileName)
    test_random_images(fileName)
    #save_label_idx_map(fileName)

