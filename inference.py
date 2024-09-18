from calendar import EPOCH
from tkinter import E
import torch
from models import Net
from torchvision import transforms
from data_load import FacialKeypointsDataset
from data_load import Normalize,Rescale,RandomCrop,ToTensor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
import numpy as np


def net_sample_output(net,test_loader):   
    net.eval() 
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):       
        # get sample data: images and ground truth keypoints
        images = sample['image'].to(device)
        key_pts = sample['keypoints'].to(device)
        # convert images to FloatTensors
        #images = images.type(torch.cuda.FloatTensor)
        images = images.type(torch.FloatTensor).to(device)
        print(images.type())
        # forward pass to get net output
        output_pts = net(images)        
        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)       
        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts
def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        if torch.is_tensor(gt_pts):
            gt_pts = gt_pts.cpu().numpy()
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10,index=0):
    plt.figure(figsize=(20,10))
    for i in range(batch_size):
        ax = plt.subplot(1, batch_size, i+1)
        
        # un-transform the image data
        image = test_images[i].data   # get the image from it's Variable wrapper
        image = image.cpu()
        image = image.numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.cpu()
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints  
        predicted_key_pts = predicted_key_pts*50.0+100
        
        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]         
            ground_truth_pts = ground_truth_pts*50.0+100
        
        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
            
        plt.axis('off')

    plt.savefig(f'./predict_index_{index}.png')
def test_net(net,test_loader,index):
    images, output_pts, gt_pts =net_sample_output(net,test_loader)
    # print out the dimensions of the data to see if they make sense
    print(images.data.size())
    print(output_pts.data.size())
    print(gt_pts.size())

    visualize_output(images, output_pts, gt_pts,batch_size=10,index=index)

if __name__ == '__main__':
    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Initialize the model
    model = Net()
    model.to(device)
    # Load the model weights
    EPOCH = 0
    model.load_state_dict(torch.load(f'/mnt/bn/ndaigc/ml_project/facial_keypoints_detect/saved_models/keypoints_model_{EPOCH}.pt', weights_only=True))
    # Define the data transforms
    data_transform = transforms.Compose([Rescale(250),
                                        RandomCrop(224),
                                        Normalize(),
                                        ToTensor()])


    transformed_testdataset = FacialKeypointsDataset(csv_file='./data/test_frames_keypoints.csv',
                                            root_dir='./data/test/',
                                            transform=data_transform)
    test_loader = DataLoader(transformed_testdataset,
                        batch_size=10,
                        shuffle=True,
                        num_workers=0)

    test_net(model,test_loader,EPOCH)