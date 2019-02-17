import os, sys, glob, pdb
import numpy as np

class Normalizer:
    def __init__(self,normalize=None):
        self.normalization = normalize
        self.images = None #original input images
        self.labels = None #original input labels

    def normalize(self,inputImg):
        '''
        Method to normalize image using predtermined normalization factor
        input: ndarray
        output: ndarray (normalized)
        '''
        nonZero = np.ma.masked_equal(inputImg,0)
        normalized = ((nonZero-self.normalization["means"])/self.normalization["vars"]).data
        return normalized

    def denormalize(self,inputImg):
        '''
        Method to apply inverse normalization to image
        input: ndarray
        output: ndarray
        '''
        #first remove unused dims
        inputImg = inputImg.squeeze()
        nonZero = np.ma.masked_equal(inputImg,0)
        image = (nonZero*self.normalization["vars"]+self.normalization["means"]).data
        
        return image

    def standardize_nii(self,niiObject,minHU=-1000, maxHU=3000):
        '''
        Pre-process images by clipping outliers and shifting -1000 to 0
        Input: nibabel object (512,512,num_slices)
        Output: normalized ndarray  
        '''
        #1. convert nii object to ndarray
        img = niiObject.get_data()

        #2. standardize dimension from [H,W,N] to [N,H,W] 
        img = np.transpose(img,[2,0,1])

        #3. nii already converted to HU's
        imgClip = (img.clip(minHU,maxHU)+1000).astype('uint16')

        #4. save original image
        self.images = imgClip

        #5. images are rotated vertically correct them here
        imgRot = np.rot90(imgClip,k=3,axes=(1,2)); 

        return imgRot

    def de_standardize_nii(self,imgInput):
        '''
        Method to convert nii images back to original format
        inputs: ndarray
        output: ndarray 
        '''
        #1. rotate back image as above
        imgRot = np.rot90(imgInput,k=-3,axes=(1,2))    
        #2. convert [N,H,W] -> [H,W,N]
        #imgRot = np.transpose(imgRot,[1,2,0])
        
        return imgRot
    
    def standardize_nii_label(self,label):
        '''
        Pre-process labels by rotating horizontally    
        Input: niibabel object (512,512,num_slices)
        Output: standardized ndimage 
        '''
        #1. convert nii to ndarray
        label = label.get_data()

        #2. standardiza dimension from [H,W,N] to [N,H,W] 
        label = np.transpose(label,[2,0,1])

        #3. save original labels
        self.labels = label

        #4. rotate label counter-clockwise
        labelRot = np.rot90(label,k=3,axes=(1,2)) 

        return labelRot

    def standardize_dicom(self,imgSlice,minHU=-1000, maxHU=3000):
        '''
        Converts pixel values to CT values, then shifts everything by 1000 to make air zero
        Input: pydicom object
        Outpur: ndarray    
        '''
        #1. convert pixel data to HU
        slope = imgSlice.RescaleSlope
        intercept = imgSlice.RescaleIntercept
        imgClip = imgSlice.pixel_array*slope+intercept

        #print("Before clipping ",sliceHU.max(), sliceHU.min(), sliceHU.dtype);
        #2 clip HU between [-1000, 3000]
        imgClip = (imgClip.clip(minHU,maxHU)+1000).astype('float32')

        #3. save original image
        self.images = imgClip

        return imgClip

    def standardize_dicom_label(self,labelMask,removeBody=False, organToSegment=False):
        '''
        Method to standardize labels; such as removing body label or binarizing labels
        '''
        if organToSegment:
            labelMask[labelMask!=organToSegment] = 0;
            labelMask[labelMask==organToSegment] = 1;

        if removeBody:
            #1. find non zero mask
            nonAir = ~(labelMask==0);    
            #2. subtract one from index, 
            labelMask[nonAir] = labelMask[nonAir]-1; 
    
        #3. save original labels
        self.labels = labelMask

        return labelMask.astype("float32")
