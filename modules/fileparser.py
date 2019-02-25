#package specific imports
from modules import *

#module specific imports
from PIL import Image
from skimage.draw import polygon
from modules.processor import ImageProcessor


class FeatureLabelReader:
    def __init__(self,pathToFiles=None, structNameMap=None):
        self.pathToFiles = os.path.abspath(pathToFiles)
        self.processor = ImageProcessor()
        self.structNameMap = structNameMap
        
    def end_point_parser(self,filePath=None,filePattern=None,dirPattern=None):
        '''
        Method to identify images contents of the directory
        filePath: directory name
        filePattern: file name to search
        dirPattern: directory to search
        output: list of file names
        '''
        #convert absolute path 
        #filePath = os.path.abspath(filePath)

        if filePath is None:
            filePath = self.pathToFiles
        else:
            self.pathToFiles = filePath

        #find endpoints
        endPoints = [subdir for subdir,dirs,files in os.walk(filePath) if len(dirs)== 0]

        if dirPattern is not None:
            #search files based on directory name
            dirMatch = []
            for path in endPoints:
                match = True
                for patt in dirPattern:
                    if patt not in path:
                        match = False
                        break
                if match:
                    dirMatch.append(path)

            #update endpoints
            endPoints = dirMatch.copy()

        if filePattern is not None:
            #find files matching description
            files = [file for path in endPoints
                    for file in glob.glob('/'.join([path,filePattern]))]
        else:
            #find files matching description
            files = ['/'.join([path,file]) for path in endPoints
                    for file in os.listdir(path)]

        #return directory names
        return sorted(files)

    def parse_images(self,imgBatch):
        '''
        Method to read file and return numpy array
        input: list of files
        output: ndarray
        '''
        N = imgBatch.__len__()
        #check file extension
        outBatch = np.zeros((N,512,512))
        for idx in range(N):
            file = imgBatch[idx]
            print(file)
            #check file extension
            if 'dcm' in file:
                img = pydicom.read_file(file)
            elif 'nii' in file:
                #TODO: Fix this part, 
                img = nib.load(file)
                return self.processor.standardize_img(img)
            else:
                img = np.array(Image.open(file))

            outBatch[idx] = self.processor.standardize_img(img)
        
        return outBatch

    def parse_labels(self,imgBatch):
        '''
        Method to read file and return numpy array
        input: list of files
        output: ndarray
        '''
        N = imgBatch.__len__()
        #check file extension
        outBatch = np.zeros((N,512,512))
        for idx in range(N):
            file = imgBatch[idx]
            print(file)
            #check file extension
            if 'dcm' in file:
                img = pydicom.read_file(file)
            elif 'nii' in file:
                #TODO: Fix this part, 
                img = nib.load(file)
                return self.processor.standardize_label(img)
            else:
                img = np.array(Image.open(file))

            outBatch[idx] = self.processor.standardize_label(img)
        
        return outBatch

    def compile_dataset(self, imgFiles=None, labelFiles=None, fileName = "TRAIN.h5", batchSize=64):
        '''
        Method to pair img-label pairs; if number of images is same as number of labels. Then they have been split
        Otherwise, labels are stored in RS file
        '''
        if (imgFiles is None) or (labelFiles is None):
            sys.exit(type(self).__name__+".compile_dataset needs both label and img files")

        if len(imgFiles)==len(labelFiles):
            #pair each image with corresponding label
            N = len(imgFiles)

            for idx in range(0,N,batchSize):
                features = self.parse_images(imgFiles[idx:idx+batchSize])
                labels = self.parse_labels(labelFiles[idx:idx+batchSize])
                print(np.unique(labels))
                print("Saving batch ",idx)
                self.save_image_mask(features, labels, fileName)

        else:
            #label mask is not seperated, so decompose RS file
            #first find mapping from RS to CT
            structImgMap = self.map_slice_to_structure(imgFiles,labelFiles)
            for key in structImgMap:
                
                #compile RS file
                structure = pydicom.read_file(key)
                contours = self.processor.dicom_rs_reader(structure,self.structNameMap)
                #compile CT slices
                slices = [pydicom.read_file(dcm) for dcm in structImgMap[key]]
                slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
           
                #combine slices and RS
                features,labels = self.create_contour_mask(contours,slices)
                print("Saving RS file ",key)
                self.save_image_mask(features, labels, fileName)

               #for img in structImgMap[key]:
               #    print("\t\t",img)


    def create_contour_mask(self,contours,slices):

        imgData = np.stack([self.processor.standardize_dicom(s) for s in slices], axis=-1)
        imgData = np.transpose(imgData,[2,0,1])
        zSlices = [float('{0:.3f}'.format(s.ImagePositionPatient[2])) for s in slices]
        zStruct = sorted(contours.keys())

        #information about patient positioning
        posR = slices[0].ImagePositionPatient[1]
        spacingR = slices[0].PixelSpacing[1]
        posC = slices[0].ImagePositionPatient[0]
        spacingC = slices[0].PixelSpacing[0]

        #imaging size
        imgRows = slices[0].Rows
        imgColumns = slices[0].Columns
        imgBits = slices[0].BitsStored

        #create image mask to store data
        ax = None
        organSlices = np.zeros((len(zSlices)),dtype=np.bool)

        imgLabel = np.zeros((len(zSlices),imgRows,imgColumns),dtype=np.uint8)
        imgBody = np.zeros((len(zSlices),imgRows,imgColumns),dtype=np.uint8)
        for zIndex, zDistance in enumerate(zSlices):
            print("Masking contours ",list(contours[zDistance].keys()))
            #access contours in given slice
            for organ in contours[zDistance]:
                for contour in contours[zDistance][organ]:
                    r = (contour[:, 1] - posR) / spacingR
                    c = (contour[:, 0] - posC) / spacingC
                    rr, cc = polygon(r, c)
                    if organ!=1:
                        imgLabel[zIndex, rr, cc] = organ
                    else:
                        imgBody[zIndex, rr, cc] = 1

            #post process current mask
            self.post_process_mask(imgBody[zIndex,...],imgLabel[zIndex,...])

            #ax = debug_image_slice(imgLabel[zIndex,...],imgData[zIndex,...],ax);
            #if slice contains only body, increment start idx    
            if np.max(imgLabel[zIndex,...])>1:
                organSlices[zIndex] = True

        #return imgData[organSlices,...],imgLabel[organSlices,...]
        return imgData,imgLabel


    def post_process_mask(self,bodyMask,labelMask):
        """
        Convert anything that is not labels to something else
        """
        newIndex = 1
        #no organ has been labeled, then label everything inside body as body
        if labelMask.max()==0:
            labelMask[bodyMask==1] = newIndex
        else:
            #there has been some organs labeled, label anything that is inside body and not organ as other, 
            undefinedMask = (labelMask==0)
            mask = bodyMask * undefinedMask
            labelMask[mask.astype('bool')] = newIndex
        # return labelMask;

    def map_slice_to_structure(self,imgFiles,labelFiles):
        '''
        Method to map strucure files to CT slices (May need optimization)
        imgFiles: list of CT images
        labelFiles: list of labels
        output: dictionary of label files and corresponding CT
        '''
        #find common path
        patientPath = set()
        
        #now iterate over each label and find common path for each img
        for label in labelFiles:
            maxMatch = 0; bestMatch = None;
            for img in imgFiles:
                currPath = os.path.commonpath([label,img])
                if len(currPath)>maxMatch:
                    maxMatch = len(currPath)
                    bestMatch = currPath
            patientPath = patientPath.union([bestMatch])
                    
        #create mapping from patient list to labels
        patToLabelMap = {pat:label for pat in patientPath for label in labelFiles if pat in label}
        
        #now map each image to label
        labelToImgMap = {}
        for key in patToLabelMap:
            currLabel = patToLabelMap[key]
            for img in imgFiles:
                if key in img:
                    if currLabel not in labelToImgMap:
                        labelToImgMap[currLabel] = []
                    else:
                        labelToImgMap[currLabel].append(img)

        return labelToImgMap

    @staticmethod
    def save_image_mask(features, labels, fileName):

        #create a hdf file to store organ
        if os.path.exists(fileName): #open file to modify
            print("Appending to dataset ... ")

            #create dataset to store images and masks
            with h5py.File(fileName, "r+") as organFile:

                #get original dataset size
                nFeatures = organFile["features"].shape[0];
                nLabels = organFile["labels"].shape[0];
                
                #resize dataset size
                organFile["features"].resize(nFeatures+features.shape[0],axis=0);
                organFile["labels"].resize(nLabels+labels.shape[0],axis=0);

                #now append new data to the end of hdf file
                organFile["features"][-features.shape[0]:] = features;
                organFile["labels"][-labels.shape[0]:] = labels;

                assert(organFile["features"].shape==organFile["labels"].shape)

        else: #create new file
            print("Creating new dataset...");

            #create dataset to store images and masks
            with h5py.File(fileName, "w") as organFile:
                organFile.create_dataset("features", data=features, maxshape=(None,)+features.shape[1:], dtype=np.uint16);
                organFile.create_dataset("labels", data=labels, maxshape=(None,)+labels.shape[1:], dtype=np.uint8);

        return fileName
