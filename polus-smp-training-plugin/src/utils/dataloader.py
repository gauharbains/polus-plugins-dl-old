import os, typing
from pathlib import Path
import numpy as np
import filepattern
import torch 
import torchvision
import segmentation_models_pytorch as smp
from bfio import BioReader, BioWriter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

class LocalNorm(object):
    def __init__(self,
                 window: int = 129,
                 clip: typing.Union[int,float] = 6):
        assert window % 2 == 1, 'window_size must be an odd integer'
        self.window = window
        self.clip = clip
        self.pad = torchvision.transforms.Pad(window//2+1,
                                              padding_mode='reflect')
        # Mode can be test, train, and eval
        self.mode = 'eval'


    def __call__(self,x):
        image = self.pad(x)
        response = self.local_response(image)
        response[response < -self.clip] = -self.clip
        response[response > self.clip] = self.clip
        return response


    def imboxfilt(self,image):
        """imboxfilt Use a box filter on a stack of images
        This method applies a box filter to an image. The input is assumed to be a
        4D array, and should be pre-padded. The output will be smaller by
        window_size-1 pixels in both width and height since this filter does not pad
        the input to account for filtering.
        """
        # Generate an integral image
        image_ii = image.cumsum(-1).cumsum(-2)
        # Create the output
        output = image_ii[...,0:-self.window-1,0:-self.window-1] + \
                 image_ii[...,self.window:-1,self.window:-1] - \
                 image_ii[...,self.window:-1,0:-self.window-1] - \
                 image_ii[...,0:-self.window-1,self.window:-1]
        return output

        
    def local_response(self,image):
        """local_response Regional normalization
        This method normalizes each pixel using the mean and standard deviation of
        all pixels within the window_size. The window_size parameter should be
        2*radius+1 of the desired region of pixels to normalize by. The image should
        be padded by window_size//2 on each side.
        """
        local_mean = self.imboxfilt(image)/(self.window ** 2)
        local_mean_square = self.imboxfilt(image.pow(2))/(self.window ** 2)
        # Use absolute difference because sometimes error causes negative values
        local_std = (local_mean_square - local_mean.pow(2)).abs().sqrt()
        local_std[local_std<10**-3] = 10**-3
        response = (image[...,
                          self.window//2:-self.window//2-1,
                          self.window//2:-self.window//2-1] - local_mean)/local_std
        return response

class Dataset(BaseDataset):
    
    def __init__(
            self, 
            images_dir, 
            labels_dir,
            images_pattern,
            labels_pattern
    ):

        self.tile_size = 256
        self.images_fp = filepattern.FilePattern(file_path=images_dir, pattern=images_pattern)
        self.labels_fp = filepattern.FilePattern(file_path=labels_dir, pattern=labels_pattern)
        self.get_tile_mapping()
        self.get_image_labels_mapping()
        self.preprocessing = torchvision.transforms.Compose([
                             torchvision.transforms.ToTensor(),
                             LocalNorm()])

    def __getitem__(self, i):
        
        
        image_name = self.tile_map[i][0]
        x, x_max = self.tile_map[i][1]
        y, y_max = self.tile_map[i][2]

        # read and preprocess image
        with BioReader(image_name) as br:
            img = br[y:y_max,x:x_max,0:1,0,0][:,:,0,0,0]
        img = img.astype(np.float32)
        img = self.preprocessing(img).numpy()

        # read and preprocess label
        label_name = self.name_map[image_name]
        with BioReader(label_name) as br:
            label = br[y:y_max,x:x_max,0:1,0,0][:,:,0,0,0]      
        label[label>=1] = 1
        label[label<1] = 0
        label = label.astype(np.float32)
        label = label.reshape((1,label.shape[0], label.shape[1]))

        return img, label
        
    def __len__(self):
        return len(self.tile_map.keys())
    
    def get_tile_mapping(self):
        """ creates a tile map for the __getitem__ function
        This function iterates over all the files in the input 
        collection and creates a dictionary that can be used in 
        __getitem_ function. 
        """
        self.tile_map = {}
        tile_size = self.tile_size
        tile_num = 0

        # iterate over all files
        for f in self.images_fp():
            file_name = f[0]['file']
            
            with BioReader(file_name) as br:
                # iterate over tiles
                for x in range(0,br.X,tile_size):
                    x_max = min([br.X,x+tile_size])
                    for y in range(0,br.Y, tile_size):
                        y_max = min([br.Y,y+tile_size])

                        # add tile to tile_map
                        self.tile_map[tile_num] = (file_name, (x,x_max), (y,y_max))
                        tile_num+=1
    
    def get_image_labels_mapping(self):
        """creates a filename map between images and labels
        In the case where image filenames have different filename 
        pattern than label filenames, this function creates a map
        between the corresponding images and labels
        """
        self.name_map = {}

        for f in self.images_fp():
            image_name = f[0]['file']
            vars = {k.upper():v for k,v in f[0].items() if k!='file' }
            label_name = self.labels_fp.get_matching(**vars)[0]['file']
            self.name_map[image_name] = label_name






        


    