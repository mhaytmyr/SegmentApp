
import cv2,numpy as np
from skimage.transform import resize
from keras.utils.np_utils import to_categorical

from modules.config import *

class Cropper:
    def __init__(self):
        self.rows = None
        self.cols = None

    def crop(self,imgInput):
        '''
        Method to crop image
        input: ndarray
        returns: list of images
        '''
        #first check dimensions of image
        if len(imgInput.shape)==2:
            rows, cols = body_bounding_box(imgInput)
            #update attributes            
            self.rows = [rows]
            self.cols = [cols]

            #crop image 
            return [crop_image_roi(imgInput,rowMin=rows[0],rowMax=rows[1],colMin=cols[0],colMax=cols[1])];

        elif len(imgInput.shape)==3:
            rows, cols , cropImg = [],[],[]
            N = imgInput.shape[0]
            
            for idx in range(N):
                row,col = body_bounding_box(imgInput[idx])
                rows.append(row)
                cols.append(col)
                #assign values to zoom
                tmp = crop_image_roi(imgInput[idx],rowMin=row[0],rowMax=row[1],colMin=col[0],colMax=col[1])
                cropImg.append(tmp)

            #update atributes
            self.rows = rows
            self.cols = cols
            
            #return cropped image
            return cropImg
        else:
            sys.exit("Error encountered in BoundingBoxCropper.bounding_box_util!")  
    
    def crop_label(self,labelInput):
        '''
        Method to crop center of label using self.cols and self.rows locations
        labelInput: label ndarray with (N,H,W,C) format
        output: list of cropped labels
        '''
        N = labelInput.shape[0]
        cropLabel = []
        for idx in range(N):
            row = self.rows[idx]
            col = self.cols[idx]
            #assign values to zoom
            tmp = crop_image_roi(labelInput[idx],rowMin=row[0],rowMax=row[1],colMin=col[0],colMax=col[1])
            cropLabel.append(tmp)
        return cropLabel

        
    def uncrop_image(self,row,col,imgInput):
        '''
        Method to pad cropped image with zeros
        input: 2D array
        output: 2D array
        '''
        rowPad = (row[0],H0-row[1])
        colPad = (col[0],W0-col[1])
        padImg = np.pad(imgInput,pad_width=(rowPad,colPad),mode='constant'); 
        return padImg

    def uncrop(self,imgInput):
        '''
        Method to pad image with zeros to return 512x512 resolution
        input: list (images have non-uniform shape so you can't concatenate)
        output: ndarray
        '''
        n = len(imgInput)
        unCropImg = np.zeros((n,H0,W0))
        for idx in range(n):
            col = self.cols[idx]
            row = self.rows[idx]
            unCropImg[idx] = self.uncrop_image(row,col,imgInput[idx])
            
        return unCropImg

    def zoom_image(self,imgInput,height=W,width=H):
        '''
        Method to zoom image, using Lancsoz interpolation over 8x8 pixel neighborhood. default: liner was not good
        input: 2d array
        returns: 2d array
        '''          
        zoomImg = cv2.resize(imgInput,(width,height),interpolation=cv2.INTER_LANCZOS4)
        return zoomImg

    def unzoom_image(self,row,col,imgInput):
        '''
        Method to dezoom image
        input: 2d array
        returns: 2d array
        '''
        deZoomImg = cv2.resize(imgInput.astype('float32'),(col[1]-col[0],row[1]-row[0]),interpolation=cv2.INTER_LANCZOS4)
        return deZoomImg

    def unzoom(self,zoomImg):
        '''
        Method to dezoom images in current batch
        To avoid ambuigity images of the current batch will be automatically used    
        '''
        unZoomImg = []
        for idx in range(zoomImg.shape[0]):
            col = self.cols[idx]
            row = self.rows[idx]
            tmp = self.unzoom_image(row,col,zoomImg[idx])
            unZoomImg.append(tmp)
        return unZoomImg 

    def unzoom_label(self,zoomLabel):
        '''
        Method to unzoom labels. Input is 4D array, so I need to zoom each label 
        separately. Then combine them. Can we optimize thi method???
        '''
        unZoomLabel = []
        for idx in range(zoomLabel.shape[0]):
            col = self.cols[idx]
            row = self.rows[idx]
            currLabel = np.zeros((row[1]-row[0],col[1]-col[0],zoomLabel.shape[-1]))
            for label in range(NUMCLASSES):
                #currLabel = cv2.resize(zoomLabel[idx],(col[1]-col[0],row[1]-row[0]),interpolation=cv2.INTER_NEAREST)
                tmp = cv2.resize(zoomLabel[idx,...,label],(col[1]-col[0],row[1]-row[0]),interpolation=cv2.INTER_LANCZOS4)
                currLabel[...,label] = tmp
            unZoomLabel.append(currLabel)
        return unZoomLabel

    def uncrop_label(self,cropLabel):
        '''
        Method to padd labels. Input is 4D array, so I need to pad each label 
        separately. Then combine them. Can we optimize thi method???
        Input: list()
        Output: 4D array
        '''
        unCropLabel = np.zeros((len(cropLabel),H0,W0,NUMCLASSES))
        for idx in range(len(cropLabel)):
            col = self.cols[idx]
            row = self.rows[idx]
            currLabel = np.zeros((H0,W0,NUMCLASSES))
            for label in range(NUMCLASSES):
                tmp = self.uncrop_image(row,col,cropLabel[idx][...,label])
                currLabel[...,label] = tmp
            unCropLabel[idx] = currLabel
        return unCropLabel

    def zoom_label(self,labelInput,height=W,width=H):
        '''
        Method to zoom batch of labels. 
        input: can be list or ndarray
        returns: 4D array (N,H,W,C)
        '''
        if type(labelInput)==np.ndarray:
            n = labelInput.shape[0]
        elif type(labelInput)==list:
            n = len(labelInput)
        else:
            sys.exit(type(self).__name__+'.zoom_label() accepts list or nd array'+type(labelInput)+' provided');

        zoomedLabel = np.zeros((n,height,width,NUMCLASSES))
        #cropImg is list so, need to iterate 
        for idx in range(n):
            #col = self.cols[idx]
            #row = self.rows[idx]
            row,col = labelInput[idx].shape

            #convert each image to categorical
            labelOneHot = to_categorical(labelInput[idx],num_classes=NUMCLASSES).reshape((row,col,NUMCLASSES))
            #labelOneHot = to_categorical(labelInput[idx],num_classes=NUMCLASSES).reshape((row[1]-row[0],col[1]-col[0],NUMCLASSES))
            for label in range(NUMCLASSES):
                zoomedLabel[idx,...,label] = cv2.resize(labelOneHot[...,label],(width,height),interpolation=cv2.INTER_LANCZOS4)

        return zoomedLabel

    def zoom(self,imgInput,height=W,width=H):
        '''
        Method to zoom batch of images
        input: can be list or ndarray
        returns: ndarray
        '''
        if type(imgInput)==np.ndarray:
            n = imgInput.shape[0]
        elif type(imgInput)==list:
            n = len(imgInput)
        else:
            sys.exit(type(self).__name__+'.zoom() accepts list or nd array'+type(imgInput)+' provided');

        zoomedImg = np.zeros((n,height,width))
        #cropImg is list so, need to iterate 
        for idx in range(n):
            tmp = self.zoom_image(imgInput[idx],height,width)
            zoomedImg[idx] = tmp; 
        
        return zoomedImg


def body_bounding_box(img):
    '''
    Method to automatically create bounding-box around the body of the ct slice. Here is cropping is done
        1. Find mass center of the image in rx, ry axis
        2. Binarize image using 200 threshold
        3. Apply morpholical opening to the top half of image
            Apply morphological eroding to the bottom half of the image
        4. calculate first non-zero location in both directions         

    Input: 2D ndarray
    Output: tuple of coordinates (rowMin, rowMax), (colMin, colMax)    
    '''

    #define kernel
    SIZE1 = 11; SIZE2 = 21;    
    kernel1 = np.ones((SIZE1,SIZE1),np.uint8)
    kernel2 = np.ones((SIZE2,SIZE2),np.uint8)

    #calculate center points
    row_mean = img.mean(axis=1)
    rows = np.arange(img.shape[0])
    row_center = np.int((rows*row_mean).sum()//row_mean.sum())
    # col_mean = img.mean(axis=0)
    # cols = np.arange(img.shape[1])
    # col_center = np.int((cols*col_mean).sum()//col_mean.sum())

    #create label img
    label = np.zeros((W0,H0),dtype=np.uint8)
    body = label.copy()

    #first binarize image
    label[img>600] = 1

    #add white spot at the center
    #img[row_center-5:row_center+5,col_center-5:col_center+5] = img.max();

    #apply morphology
    body[:row_center] = cv2.morphologyEx(label[:row_center], cv2.MORPH_OPEN, kernel1)
    body[row_center:] = cv2.morphologyEx(label[row_center:], cv2.MORPH_ERODE, kernel2)

    #compute non zeros
    rows = np.any(body, axis=1)
    cols = np.any(body, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    #apply margin
    ymin, ymax = max(0,ymin-SIZE1//2), min(ymax+SIZE2//2,512)
    xmin, xmax = max(0,xmin-SIZE1//2), min(xmax+SIZE1//2,512) 
    
    #return crop positions
    return (ymin,ymax),(xmin,xmax)

def crop_image_roi(img,rowMin=ROW,rowMax=ROW+W,colMin=COL,colMax=COL+H):
    '''
    Method to crop center of image using pre-defined regions. Not using it anymore
    Input: can be 2D, 3D (may have bug, assumes [N,H,W]) or 4D image
    Output: 
    '''
    if len(img.shape)==2:
        return img[rowMin:rowMax,colMin:colMax]
    elif len(img.shape)==3:
        return img[:,rowMin:rowMax,colMin:colMax]
    elif len(img.shape)==4:
        return img[:,rowMin:rowMax,colMin:colMax,:]
    else:
        sys.exit("preprocessing.crop_image_roi method can't crop img size of",img.shape)
        
