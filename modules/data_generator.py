# import numpy as np, sys, pdb
# import cv2, h5py, dask
# import dask.array as da
# import dask
# import time

#package specific imports
from modules import *

#module specific imports
from queue import Queue
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from modules.processor import ImageProcessor


class DataGenerator(ImageProcessor):
    def __init__(self,normalize=None):
        ImageProcessor.__init__(self,normalize)
        self.augments = []
        
    def load_file(self,inputFile):
        '''
        Input file format is assumed to be hd5
        TODO: add methods to process different inputs
        '''
        filePntr = h5py.File(inputFile,'r')
        return filePntr

    def augment_img_label(self,imgInput,labelInput):
        '''
        Method to augment data and labels using horizontal/veritcal flipping and elastic deformation 
        imgInput: CT image after pre-processing and normalization shape=(N,H,W)
        labelInput: segmentation mask, shape=(N,H,W,C)
        '''
        #copy image to memory, hdf5 doesn't allow inplace
        #image are already loaded to memory using dask.array
        imgNew = imgInput.copy()
        labelNew = labelInput.copy()

        #get image info
        n,row,col = imgInput.shape
        #keep track of augmentation
        augments = []

        for idx in range(n):	
            choice = np.random.choice(['flip','nothing','zoom','zoom','nothing'])
            augments.append(choice)
            if choice=='flip':
                imgNew[idx,...] = imgInput[idx,:,::-1]
                labelNew[idx,...] = labelInput[idx,:,::-1,:]
            elif choice=='rotate':
                imgNew[idx,...] = imgInput[idx,::-1,:]
                labelNew[idx,...] = labelInput[idx,::-1,:,:]
            elif choice=='zoom':
                zoomfactor = np.random.randint(11,25)/10
                dx = np.random.randint(-20,20)
                dy = np.random.randint(-20,20)
                M_zoom = cv2.getRotationMatrix2D((row/2+dx,col/2+dy), 0, zoomfactor)
            
                imgNew[idx,...] = cv2.warpAffine(imgInput[idx,...], M_zoom,(col,row))
                if len(labelInput.shape)==4:
                    for label in range(labelInput.shape[-1]):
                        labelNew[idx,...,label] = cv2.warpAffine(labelInput[idx,...,label], M_zoom,(col,row))

            elif choice=='deform':
                #draw_grid(imgNew[idx,...], 50)
                #draw_grid(organNew[idx,...], 50)
                #combine two images
                merged = np.dstack([imgInput[idx,...], labelInput[idx,...]])
                #apply transformation
                mergedTrans = elastic_transform(merged, merged.shape[1] * 3, merged.shape[1] * 0.08, merged.shape[1] * 0.08)
                #now put images back
                imgNew[idx,...] = mergedTrans[...,0]
                labelNew[idx,...] = mergedTrans[...,1:]

        #update augmentation attribute, to restore original data
        self.augments = augments

        return imgNew, labelNew

    def generate_data(self,inputFile=None,batchSize=16,augment=False,shuffle=False):
        #get file
        hdfFile = self.load_file(inputFile)
        
        #initialize pointer
        idx,n = 0, hdfFile["features"].shape[0]
        indices = np.arange(n)

        if shuffle:
            np.random.shuffle(indices)

        while True:
            start = idx
            end = (idx+batchSize)
        
            if idx>=n:
                #shuffle indices after each epoch
                if shuffle: 
                    np.random.shuffle(indices)

                slice = np.arange(start,end)
                subIndex = sorted(indices[slice%n])
                idx = end%n

                #get data    
                imgBatch = hdfFile["features"][subIndex,...]
                labelBatch = hdfFile["labels"][subIndex,...]
            else:
                #increment counter
                idx+=batchSize

                if shuffle:
                    subIndex = sorted(indices[start:end])
                    imgBatch = hdfFile["features"][subIndex,...]
                    labelBatch = hdfFile["labels"][subIndex,...]
                else:
                    imgBatch = hdfFile["features"][start:end,...]
                    labelBatch = hdfFile["labels"][start:end,...]

            #convert to one-hot encoded
            feature, organ = self.pre_process_img_label(imgBatch,labelBatch)

            #augment data
            if augment:
                feature,organ = self.augment_img_label(feature,organ)

            #create generator
            yield (self.img_to_tensor(feature),{'organ_output':self.img_to_tensor(organ)})


    def data_generator_stratified(self,hdfFileName,batchSize=50,augment=True,normalize=None):
        '''
        Method to generate data with balanced class in each batch
        TODO: fix this
        '''

        #create place holder for image and label batch
        img_batch = np.zeros((batchSize,config1["H0"],config1["W0"]),dtype=np.float32)
        label_batch = np.zeros((batchSize,config1["H0"],config1["W0"]),dtype=np.float32)
    
        #get pointer to features and labels
        hdfFile = h5py.File(hdfFileName,"r")
        features = hdfFile["features"]        
        labels = hdfFile["labels"]

        #create dask array for efficienct access    
        daskFeatures = dask.array.from_array(features,chunks=(4,config1["H0"],config1["W0"]))
        daskLabels = dask.array.from_array(labels,chunks=(4,config1["H0"],config1["W0"]))

        #create queue for keys
        label_queue = Queue()
            
        #create dictionary to store queue indices
        label_idx_map = {}
        #(no need to shuffle data?), add each index to queue
        with h5py.File(hdfFileName.replace(".h5","_IDX_MAP.h5"),"r") as fp:
            for key in fp.keys():
                label_queue.put(key)
                label_idx_map[key] = Queue()
                for item in fp[key]:
                    label_idx_map[key].put(item)

        #yield batches
        while True:
            #start = time.time()
            for n in range(batchSize):
                #get key from keys queue
                key = label_queue.get()
                #get corresponding index
                index = label_idx_map[key].get();            
                #append them to img_batch and label_batch
                img_batch[n] = daskFeatures[index].compute()
                label_batch[n] = daskLabels[index].compute()

                #circulate queue
                label_queue.put(key)
                label_idx_map[key].put(index)

            #debug queue
            #print("{0:.3f} msec took to generate {1} batch".format((time.time()-start)*1000,batchSize))
            #print(label_idx_map["2"].queue);

            #apply pre-processing operations
            feature, organ = self.pre_process_img_label(img_batch,label_batch)

            #augment data
            if augment:
                feature,organ = augment_data(feature,organ)

            #yield data 
            #yield (feature[...,np.newaxis], {'organ_output':organ})
            yield (self.img_to_tensor(feature),{'organ_output':self.img_to_tensor(organ)})

# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(1,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(1,))


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)







