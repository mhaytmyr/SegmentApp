#package specific imports
from modules import *

#module specific imports
from queue import Queue
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from modules.processor import ImageProcessor


class DataGenerator(ImageProcessor):
    def __init__(self,normalize=None,crop=True):
        ImageProcessor.__init__(self,normalize)
        self.crop = crop
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
            feature, organ = self.pre_process_img_label(imgBatch,labelBatch,crop=self.crop)
            

            #augment data
            if augment:
                feature,organ = self.augment_img_label(feature,organ)

            #create generator
            yield (self.img_to_tensor(feature),{'organ_output':self.img_to_tensor(organ)})

    def data_generator_discriminator(self,inputFile=None,batchSize=16,shuffle=False, create_fake=True):
        '''
        Method to generate fake data by randomly merging datasets
        '''
        #get file
        hdfFile = self.load_file(inputFile)
        print(hdfFile["features"].shape, hdfFile["labels"].shape)

        #initialize pointer
        idx,n = 0, hdfFile["features"].shape[0]
        #get only first 10 images
        idx,n = 120,124
        indices = np.arange(n-idx)+idx

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
                imgBatch = hdfFile["features"][subIndex,...]
                
            else:
                #increment counter
                idx+=batchSize

                if shuffle:
                    subIndex = sorted(indices[start:end])
                    imgBatch = hdfFile["features"][subIndex,...]
                else:
                    imgBatch = hdfFile["features"][start:end,...]

            #convert to one-hot encoded
            feature = self.pre_process_img(imgBatch,crop=self.crop)
            organ = np.zeros(feature.shape[0])

            #create fake images for discriminator training
            if create_fake:
                #now distort half of images to create fake dataset
                for index in range(feature.shape[0]//2):
                    tmp = feature[index,...]
                    distorted = elastic_transform_2d(tmp, tmp.shape[1] * 3, tmp.shape[1] * 0.08, tmp.shape[1] * 0.08)
                    #now put images back
                    feature[index,...] = distorted
                    organ[index] = 1

            #create generator
            yield (self.img_to_tensor(feature),{'out':organ})

# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(1,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(1,))

def elastic_transform_2d(image, alpha, sigma, alpha_affine, random_state=None):
    """
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)



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







