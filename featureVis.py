import torch
from models import Net
from torchvision import transforms
from data_load import FacialKeypointsDataset
from data_load import Normalize,Rescale,RandomCrop,ToTensor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2 



if __name__ == '__main__':
    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Initialize the model
    model = Net()
    model.to(device)
    # Load the model weights
    EPOCH = 200
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
    # Set the model to evaluation mode
    model.eval()
    # Get the weights in the first conv layer, "conv1"
    # if necessary, change this to reflect the name of your first conv layer
    weights1 = model.conv4.weight.data.cpu()

    w = weights1.numpy()

    filter_index = 31

    print(w[filter_index][0])
    print(w[filter_index][0].shape)

    # display the filter weights
    plt.imshow(w[filter_index][0], cmap='gray')
    plt.savefig(f'./kernel_{filter_index}.png')
    # Define the data transforms
    data_transform = transforms.Compose([Rescale(250),
                                        RandomCrop(224),
                                        Normalize(),
                                        ToTensor()])


    transformed_testdataset = FacialKeypointsDataset(csv_file='./data/test_frames_keypoints.csv',
                                            root_dir='./data/test/',
                                            transform=data_transform)
    image= transformed_testdataset[0]['image']
    #print(image)
    image = image.cpu().numpy()
    image = np.squeeze(image)
    dst = cv2.filter2D(image, -1,w[filter_index][0])
    plt.imshow(dst, cmap='gray')
    plt.savefig(f'./featureMap_{filter_index}.png')
