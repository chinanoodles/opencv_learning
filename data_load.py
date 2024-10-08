# import the required libraries
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, utils


import cv2

def data_visualization(index):
    key_pts_frame = pd.read_csv('./data/training_frames_keypoints.csv')
    n = index
    image_name = key_pts_frame.iloc[n, 0]

    key_pts = key_pts_frame.iloc[n, 1:].values
    print(len(key_pts)/2)
    key_pts = key_pts.astype('float').reshape(-1, 2)

    print('Image name: ', image_name)
    print('Landmarks shape: ', key_pts.shape)
    print('First 4 key pts: {}'.format(key_pts[:4]))
    print('Number of images: ', key_pts_frame.shape[0])

def show_keypoints(image, key_pts):
    """Show image with keypoints"""
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')

class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    def display_samples(self,numberOfSamples):
        num_to_display = numberOfSamples
        # define the size of images
        fig = plt.figure(figsize=(20,10))
        for i in range(num_to_display):       
            # randomly select a sample
            rand_i = np.random.randint(0, len(self))
            sample = self[rand_i]
            # print the shape of the image and keypoints
            print(i, sample['image'].shape, sample['keypoints'].shape)

            ax = plt.subplot(1, num_to_display, i + 1)
            ax.set_title('Sample #{}'.format(i))
            
            # Using the same display function, defined earlier
            show_keypoints(sample['image'], sample['keypoints'])
        plt.savefig(f'./output.png')

# tranforms
class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']       
        #image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0
        return {'image': image_copy, 'keypoints': key_pts_copy}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))       
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]
        return {'image': img, 'keypoints': key_pts}
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h,
                      left: left + new_w]
        key_pts = key_pts - [left, top]
        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}

if __name__ == "__main__":
    # define the data tranform
    # order matters! i.e. rescaling should come before a smaller crop
    data_transform = transforms.Compose([Rescale(250),
                                        RandomCrop(224),
                                        Normalize(),
                                        ToTensor()])
    # create the transformed dataset
    transformed_dataset = FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                                 root_dir='./data/training/',
                                                 transform=data_transform)
    # print some stats about the transformed data
    # make sure the sample tensors are the expected size
    for i in range(5):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['keypoints'].size())