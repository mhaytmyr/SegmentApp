#package specific imports
from modules import *

#module specific imports
from collections import defaultdict
# from keras.utils.np_utils import to_categorical
from modules.normalizer import Normalizer
from modules.cropper import Cropper

class ImageProcessor(Normalizer,Cropper):
	
    def __init__(self,normalize=None):
        Normalizer.__init__(self,normalize)
        Cropper.__init__(self)

    def standardize_img(self,inputFile):
        #first convert image to numpy array
        if type(inputFile)==nib.nifti1.Nifti1Image:#input file is nii
            imgStandard = self.standardize_nii(inputFile)
        elif type(inputFile)==pydicom.dataset.FileDataset:#input file is dcm
            imgStandard = self.standardize_dicom(inputFile)
        elif type(inputFile)==np.ndarray:#input must be already standardized
            self.images = inputFile
            imgStandard = inputFile
        else:
            sys.exit(type(self).__name__+".standardize_img can't standardize inpuy file")
        return imgStandard

    def standardize_label(self,inputFile):
        #first convert image to numpy array
        if type(inputFile)==nib.nifti1.Nifti1Image:#input file is nii
            labelStandard = self.standardize_nii_label(inputFile)
        elif type(inputFile)==pydicom.dataset.FileDataset:#input file is dcm
            labelStandard = self.standardize_dicom_label(inputFile)
        elif type(inputFile)==np.ndarray:#input must be already standardized
            self.labels = inputFile
            labelStandard = inputFile
        else:
            sys.exit(type(self).__name__+".standardize_label can't standardize inpuy file")
        return labelStandard

    def img_to_tensor(self,array):
        '''
        Method to convert numpy array to 4D tensor to process in tensorflow
        Input: ndarray, must be at least 2D image
        Output: 4D ndarray
        '''
        #1. get array shape
        shape = array.shape

        if len(shape)==2:
            return array[np.newaxis,...,np.newaxis]
        elif len(shape)==3:
            #num channel exist, batch size missing
            if (np.prod(shape[1:])==config1["H0"]*config1["W0"]) or (np.prod(shape[1:])==config1["H"]*config1["W"]):
                return array[...,np.newaxis]
            #num batches exist but, channel missing
            elif (np.prod(shape[:-1])==config1["H0"]*config1["W0"]) or (np.prod(shape[:-1])==config1["H"]*config1["W"]):        
                return array[np.newaxis,...]
            else:
                sys.exit("preprocessing.img_to_tensor method can't convert ",shape," to 4D tensor");
        elif len(shape)==4:#already 4D tensor
            return array
        else:
            sys.exit("preprocessing.img_to_tensor method can't convert ",shape," to 4D tensor");

    def pre_process_img_label(self,imgBatch,labelBatch):
        '''
        Method to pre-process image and label simultanously. 
        imgBatch: standardized ndarray (CT number clipped, transposed and rotated)
        labelBatch: pre-processed label (enumerated label, transposed and rotated)
        '''
        #1. pre-processing image only
        imgNorm = self.pre_process_img(imgBatch)
        #2. pre-process label
        labelZoom = self.pre_process_label(labelBatch)

        return imgNorm, labelZoom

    def pre_process_label(self,labelInput):
        '''
        Method to pre-process label file. Zooming slightly distorts label mask, it seems effect is small (<3%)
        inputFile: .nii, dcm or ndarray
        output: cropped label mask, shape=(N,H,W,C)
        ''' 
        labelStandard = self.standardize_label(labelInput)
        labelCrop = self.crop_label(labelStandard)
        labelZoom = self.zoom_label(labelCrop)
        #convert zoomed label to categorcial, 
        #1. apply argmax, since zooming return type is double
        #2. perform one-hot encoder then reshape
        #min = labelZoom.min(axis=(1,2),keepdims=True)
        #max = labelZoom.max(axis=(1,2),keepdims=True)
        #max[max==0] = 1
        #labelZoomNorm = (labelZoom-min)/(max-min)
        labelZoomCat = to_categorical((labelZoom).argmax(axis=-1),
                        config1["NUMCLASSES"]).reshape((-1,config1["W"],config1["H"],config1["NUMCLASSES"]))
        #pdb.set_trace()

        return labelZoomCat

    def pre_process_img(self,inputFile):
        '''
        Method to pre-process input file
        inputFile: .nii or dcm file
        output: cropped and normalized ndarray, which can be directly passed to model
        ''' 
        imgStandard = self.standardize_img(inputFile) #convert to numpy
        imgCrop = self.crop(imgStandard)
        imgZoom = self.zoom(imgCrop)
        imgNorm = self.normalize(imgZoom)
        
        return imgNorm

    def inverse_pre_process_img(self,imgNorm):
        #apply inverse preprocessing
        imgDeNorm = self.denormalize(imgNorm)
        imgUnZoom = self.unzoom(imgDeNorm)
        imgDeCrop = self.uncrop(imgUnZoom)

        return imgDeCrop

    def morphological_operation(self,img,operation='close'):
        '''
        post-processing methods on label
        '''
        kernel = np.ones((5,5),np.uint8);

        if len(img.shape)==2:
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        else:
            outImg = np.zeros_like(img);
            for idx in range(img.shape[-1]):
                if operation=='close':
                    outImg[...,idx] = cv2.morphologyEx(img[...,idx], cv2.MORPH_CLOSE, kernel)
                elif operation=='open':
                    outImg[...,idx] = cv2.morphologyEx(img[...,idx], cv2.MORPH_OPEN, kernel)
                elif operation=='dilate':
                    outImg[...,idx] = cv2.morphologyEx(img[...,idx], cv2.MORPH_DILATE, kernel)
                else:
                    sys.exit('morphological operation invalid ')
            return outImg


    def dicom_rs_reader(self,structure,structNameMap):
        #create key pairs with key corresponding to z component
        contours = defaultdict(dict)
        for i in range(len(structure.ROIContourSequence)):

            #create contour data
            roiName = structure.StructureSetROISequence[i].ROIName
            roiNumber = structNameMap(roiName)
            print("Extracting contour for ",roiName)

            try:
                #get collection of contours for this label
                for s in structure.ROIContourSequence[i].ContourSequence:
                    node = np.array(s.ContourData).reshape(-1,3);
                    zIndex = float("{0:.3f}".format(node[0,2])); #convert to float for string
                    if roiNumber in contours[zIndex]:
                        contours[zIndex][roiNumber].append(node[:,:2]);
                    else:
                        contours[zIndex][roiNumber] = [node[:,:2]];
                print(contours[zIndex].keys());
            except Exception as e:
                print(e)

        #contours is dictionary of dictionaries
        return contours
