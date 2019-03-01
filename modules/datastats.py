from modules import *
from modules.processor import ImageProcessor


def save_train_stats(trainFileName):

    fp = h5py.File(trainFileName, "r")
    data = fp["features"]
    data_crop = np.zeros((data.shape[0],config1["W"],config1["H"]))
    batch = 32

    #initialize ImageProcessor class
    processor = ImageProcessor();    

    #apply bounding box preprocesing for each image before stats
    for idx in range(0,data.shape[0],batch):
        imgCrop = processor.crop(data[idx:idx+batch])
        imgZoom = processor.zoom(imgCrop)
        data_crop[idx:idx+batch] = imgZoom
        if idx%256==0:
            print("Processing batch ",idx)
    print("Completed cropping!")
    #create dask array for further processing
    data_da = da.from_array(data_crop,chunks=(4,config1["W"],config1["H"])) #parallelize

    #Mask out zero measurements, air and recompute mean and variance
    nonZero = da.ma.masked_equal(data_da,0)
    mean = nonZero.mean(axis=0,keepdims=True)
    variance = nonZero.std(axis=0,keepdims=True)
    print("Computed masked array!");

    #now store them in new file
    with h5py.File(trainFileName.replace(".h5","_STAT.h5"), "w") as newFile:
        newFile.create_dataset("means", data=mean.compute(), dtype=np.float32)
        newFile.create_dataset("variance", data=variance.compute(), dtype=np.float32)
    print("Computed variance and mean array!");


def save_label_idx_map(trainFileName):
    #get labels
    labels = h5py.File(trainFileName,"r")["labels"]
    labels_da = da.from_array(labels,chunks=(4,512,512))
    
    label_idx_map = {}    
    #count number of occurences for each label
    for idx in range(1,config1["NUMCLASSES"]):
        start = time.time()
        X,Y,Z = da.where(labels_da==idx)
        label_idx_map[idx] = da.unique(X).compute(); 
        print("Finished label {0} in {1:.3f} s".format(idx,time.time()-start));    

    with h5py.File(trainFileName.replace(".h5","_IDX_MAP.h5"),"w") as newFile:
        for idx in range(1,config1["NUMCLASSES"]):
            newFile.create_dataset(str(idx), data=label_idx_map[idx], dtype=np.int16);
    