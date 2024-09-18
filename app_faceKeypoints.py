# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
from models import Net
from data_load import Normalize,Rescale,RandomCrop,ToTensor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        
    def __call__(self, sample):
        image = sample
        #image_copy = np.copy(image)
        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        return image_copy
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
        image = sample
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
        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample 
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))        
        return torch.from_numpy(image)
class face_dataset(Dataset):
    def __init__(self, data_set, transform=None):
        self.data_set = data_set
        self.transform = transform

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        image = self.data_set[idx]        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]        
        sample = image

        if self.transform:
            sample = self.transform(sample)                
        return sample
def show_keypoints(image, key_pts):
    """Show image with keypoints"""
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')

# load in color image for face detection
image = cv2.imread('./data/myphoto/莫文蔚_2.jpg')
image = cv2.imread('./data/test/Adrian_Nastase_40.jpg')
# switch red and blue color channels 
# --> by default OpenCV assumes BLUE comes first, not RED as in many images
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # plot the image
# fig = plt.figure(figsize=(9,9))
# plt.imshow(image)
# plt.savefig('./appTst.png')

# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('./detector_architecture/haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(image, 1.2, 2)

# make a copy of the original image to plot detections on
image_with_detections = image.copy()

# loop over the detected faces, mark the image where each face is found
for (x,y,w,h) in faces:
    # draw a rectangle around each detected face
    # you may also need to change the width of the rectangle drawn depending on image resolution
    cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3) 

fig = plt.figure(figsize=(9,9))
image_copy = np.copy(image)
images_face = []
# loop over the detected faces from your haar cascade
for index,(x,y,w,h) in enumerate(faces):
    
    # Select the region of interest that is the face in the image 
    roi = image_copy[y:y+h, x:x+w]
    images_face.append(roi)


#print(len(images_face))

#Initialize the Model and Load Weights
EPOCH = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
net.load_state_dict(torch.load(f'./saved_models/keypoints_model_{EPOCH}.pt', weights_only=True))
net.eval()

data_transform = transforms.Compose([Rescale(224),
                                        Normalize(),
                                        ToTensor()])

faces_data = face_dataset(images_face,data_transform)

batch_size  = 10

test_loader = DataLoader(faces_data,
                    batch_size=10,
                    shuffle=True,
                    num_workers=0)

# print(faces_data[0])
# image = faces_data[0].cpu().numpy()   # convert to numpy array from a Tensor
# plt.imshow(np.squeeze(image),cmap="gray")
# plt.savefig(f'./faces_tst.png')
for i,face in enumerate(test_loader):
    sample = face
    image_tensor = sample.type(torch.FloatTensor).to(device)
    outputs = net(image_tensor)
    keypoints = outputs.view(outputs.size()[0], 68, -1)    
    print(image_tensor.data.size())
    print(keypoints.data.size())
    for j in range(batch_size):
        # un-transform the predicted key_pts data
        predicted_key_pts = keypoints[i].data
        predicted_key_pts = predicted_key_pts.cpu()
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints  
        #predicted_key_pts = predicted_key_pts*50.0+100

    #print(keypoints)
    # images = sample.to(device)
    # images = images.type(torch.FloatTensor).to(device)
    #print(i, image_tensor.data.size())

 
    # keypoints = keypoints*50.0+100
    # print(keypoints)
    image = image_tensor.cpu().numpy()   # convert to numpy array from a Tensor
    #image = np.transpose(image, (1, 2, 0)) 
    plt.imshow(np.squeeze(image),cmap="gray")
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    plt.savefig(f'./faces_{i}.png')



    