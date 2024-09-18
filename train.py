## TODO: Define the Net in models.py
import torch
import torch.nn.functional as F
from data_load import FacialKeypointsDataset
from data_load import Normalize,Rescale,RandomCrop,ToTensor
from models import Net
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from torch import optim


# test the model on a batch of test images

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

def train_net(n_epochs,train_loader):
    # prepare the net for training
    net.train()
    for epoch in range(n_epochs):  # loop over the dataset multiple times        
        running_loss = 0.0
        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image'].to(device)
            key_pts = data['keypoints'].to(device)
            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)
            # convert variables to floats for regression loss
            #key_pts = key_pts.type(torch.cuda.FloatTensor)
            key_pts = key_pts.type(torch.FloatTensor).to(device)
            #images = images.type(torch.cuda.FloatTensor)
            images = images.type(torch.FloatTensor).to(device)
            # forward pass to get outputs
            output_pts = net(images)
            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)
            # zero the parameter (weight) gradients
            optimizer.zero_grad()            
            # backward pass to calculate the weight gradients
            loss.backward()
            # update the weights
            optimizer.step()
            # print loss statistics
            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss = 0.0
        if epoch % 10 == 0:
            # save the model
            torch.save(net.state_dict(), 'saved_models/keypoints_model_{}.pt'.format(epoch))
            print('Model saved at epoch {}'.format(epoch))

    print('Finished Training')

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

    # Define the data transforms
    data_transform = transforms.Compose([Rescale(250),
                                        RandomCrop(224),
                                        Normalize(),
                                        ToTensor()])
    # create the transformed dataset
    transformed_dataset = FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                                    root_dir='./data/training/',
                                                    transform=data_transform)
    transformed_testdataset = FacialKeypointsDataset(csv_file='./data/test_frames_keypoints.csv',
                                                root_dir='./data/test/',
                                                transform=data_transform)
    print('Number of images: ', len(transformed_dataset))
    # iterate through the transformed dataset and print some stats about the first few samples
    # for i in range(4):
    #     sample = transformed_dataset[i]
    #     print(i, sample['image'].size(), sample['keypoints'].size())
    # Instantiate the network and move it to the device (GPU or CPU)
    net = Net()
    net.to(device)
    print(net)

    # load training data in batches
    batch_size = 10
    train_loader = DataLoader(transformed_dataset, 
                            batch_size=batch_size,
                            shuffle=True, 
                            num_workers=0)

    # Specify the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)


    n_epochs = 300
    # test the model on the test dataset
    test_loader = DataLoader(transformed_testdataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
    train_net(n_epochs,train_loader)

    index =0                        
    test_net(net,test_loader,index)


