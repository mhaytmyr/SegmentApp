# import numpy as np
# import cv2,sys
# import matplotlib.pyplot as plt

from modules import *

class Plotter:
    def __init__(self):
        pass

    def plot(self,imgInput=None,labelInput=None,predInput=None):
        if imgInput is None:
            #nothing to plot; raise error
            sys.exit(type(self).__name__+".plot() imgInput has to be provided ")
        elif labelInput is None:
            #only plot imgInput
            return self.plot_slice(imgInput)
        elif predInput is None:
            #show img and ground truth
            return self.plot_slice_label(imgInput,labelInput)
        else:
            #show both pred and ground truth
            return self.plot_label_prediction(imgInput,imgInput,labelInput)

    def plot_slices(self,imgInput):
        #imgInput is list of images
        imgStack = []
        for idx in range(len(imgInput)):
            print(type(self).__name__+".plot_slices img shape"+str(imgInput[idx].shape))
            imgStack.append(self.tensor_to_image_wrapper(imgInput[idx]))

        imgNorm = np.hstack(imgStack)
        cv2.imshow("Slice ",imgNorm)
        k = cv2.waitKey(0)
        return k

    def tensor_to_image_wrapper(self,imgInput=None,idx=None):
        #Method to convert 4D array image to 2D image
        #first remove extra dimentions
        imgInput = imgInput.squeeze()
        #if 2D image nothing to do; return image
        if len(imgInput.shape)==2:
            max = imgInput.max(keepdims=True)
            max[max==0] = 1
            imgNorm = (imgInput-imgInput.min())/max
        elif len(imgInput.shape)==3:
            if idx is None:
                idx = np.random.randint(0,imgInput.shape[0]-1)
            max = imgInput[idx].max(keepdims=True)
            #avoid zero division
            max[max==0] = 1
            imgNorm = (imgInput[idx]-imgInput[idx].min())/max
        else:
            sys.exit(type(self).__name__+".tensor_to_image_wrapper() can't convert input to image "+str(imgInput.shape)) 
        return imgNorm

    def tensor_to_label_wrapper(self,labelInput=None,idx=None):
        #Method to convert 4D label array to 2D image'
        labelInput = labelInput.squeeze()
        #apply argmax to last index, since prediction is one-hot encoded
        if len(labelInput.shape)==4:
            labelInput = labelInput.argmax(axis=-1) 
        return self.tensor_to_image_wrapper(labelInput,idx)

    def plot_slice(self,imgInput):
        #method to plot image slice only
        #first denormalize image; 
        imgNorm = self.tensor_to_image_wrapper(imgInput)
        cv2.imshow("Slice ",imgNorm)
        k = cv2.waitKey(0)
        return k

    def plot_slice_label(self,imgInput,labelInput):
        #Method to plot image and label masks
        #input is 4D array, so we need to choose one
        idx =  np.random.randint(0,imgInput.shape[0]-1)
        img = self.tensor_to_image_wrapper(imgInput,idx=idx)
        label = self.tensor_to_label_wrapper(labelInput,idx=idx)

        imgNorm = np.hstack([img,label])
        cv2.imshow("Slice ",imgNorm)
        k = cv2.waitKey(0)
        return k

    def plot_label_prediction(self,imgInput,labelInput,predInput):
        #Method to plot image label and prediction masks
        #input is 4D array, so we need to choose one
        idx =  np.random.randint(0,imgInput.shape[0]-1)
        img = self.tensor_to_image_wrapper(imgInput,idx=idx)
        label = self.tensor_to_label_wrapper(labelInput,idx=idx)
        pred = self.tensor_to_label_wrapper(predInput,idx=idx)

        imgStack = np.hstack([img,label,pred])
        cv2.imshow("Left: Slice, Middle: Ground True, Right: Prediction ",imgStack)
        k = cv2.waitKey(0)
        return k

    def overlay_contours_plot(self,pred,truth,image,imgNum,ax=None):
        '''
        Method to overlay contours
        TODO: fix
        '''
        colors = ('g','r','c','m','b','y');

        if ax is None:
            fig, axes = plt.subplots(ncols=2, figsize=(15,8), gridspec_kw = {'wspace':0, 'hspace':0,'left':0,'right':1.0,'top':0.95,'bottom':0.});
            ax = axes.ravel();

        ax[0].set_title("True contours");
        ax[0].imshow(image, cmap="gray");
        
        for idx in [1,2,3,4,5,6]: 
            ax[0].contour((truth==idx).astype('float32'), colors=colors[idx-1], normlinestyles='solid', linewidths=1);

        ax[0].axis('off');

        ax[1].set_title("Predicted contours");    
        ax[1].imshow(image, cmap="gray");
        for idx in [1,2,3,4,5,6]:
            ax[1].contour((pred==idx).astype('float32'), colors=colors[idx-1], linestyles='solid', linewidths=1);
        ax[1].axis('off');    

        #plt.tight_layout();
        plt.show(0);
        plt.pause(1);
        
        ax[1].clear(); ax[0].clear();        

        return ax

    def overlay_contours_save(pred,truth,image,imgNum):
        '''
        Method to overlay contours and save them to dcm
        TODO: fix
        '''    

        fig = plt.figure(figsize=(8, 6));
        colors = ('g','r','c','m','b','y');

        
        sub1 = fig.add_subplot(121)
        sub1.imshow(image, cmap="gray");
        
        for idx in [1,2,3,4,5,6]: 
            sub1.contour((truth==idx).astype('float32'), colors=colors[idx-1], normlinestyles='solid', linewidths=0.7);

        sub1.axis('off');

        sub2 = fig.add_subplot(122)
        sub2.imshow(image, cmap="gray");
        for idx in [1,2,3,4,5,6]:
            sub2.contour((pred==idx).astype('float32'), colors=colors[idx-1], linestyles='solid', linewidths=0.7);
        sub2.axis('off');

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.0, left=0.0, top = 1, right=1, wspace=0, hspace=0)
        plt.pause(1);

        plt.savefig('./figures/val_seg_'+str(imgNum)+".jpg", dpi=400, bbox_inches='tight');
        plt.close("all");
