# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from models import Net
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.image as mpimg


class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        
    def __call__(self, sample):
        image = sample
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)        
        image_copy = image_copy / 255.0        
        return image_copy

class Rescale(object):
    """Rescale the image in a sample to a given size."""
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
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)
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
        if image.shape[2] == 4:
            image = image[:, :, :3]        
        sample = image

        if self.transform:
            sample = self.transform(sample)                
        return sample

def show_keypoints(image, key_pts):
    """Show image with keypoints"""
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')

# Load in color image for face detection
# image = cv2.imread('./data/myphoto/Abdullah_Gul_10.jpg')

if __name__ == '__main__':

    image = cv2.imread('./data/test/Adrian_Nastase_40.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load in a Haar cascade classifier for detecting frontal faces
    face_cascade = cv2.CascadeClassifier('./detector_architecture/haarcascade_frontalface_default.xml')

    # Run the detector
    faces = face_cascade.detectMultiScale(image, 1.2, 2)

    # Make a copy of the original image to plot detections on
    image_with_detections = image.copy()

    # Loop over the detected faces, mark the image where each face is found
    for (x, y, w, h) in faces:
        cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), (255, 0, 0), 3) 

    fig = plt.figure(figsize=(9, 9))
    image_copy = np.copy(image)
    images_face = []
    # Loop over the detected faces from your Haar cascade
    for index, (x, y, w, h) in enumerate(faces):
        roi = image_copy[y:y + h, x:x + w]
        cv2.imwrite(f'./faces_tst_{index}.png', cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
        images_face.append(roi.astype(np.float32)/255.0)
    # for debug
    # for idx, img in enumerate(images_face):
    #     print(f"Image {idx} - shape: {img.shape}, dtype: {img.dtype}")
    data_transform = transforms.Compose([Rescale(224), Normalize(), ToTensor()])
    faces_data = face_dataset(images_face, transform=data_transform)
    # Create DataLoader
    batch_size = 1
    test_loader = DataLoader(faces_data,
                            batch_size=batch_size,
                            shuffle=False,  # Set to False if order matters
                            num_workers=0)
    # Initialize the Model and Load Weights
    EPOCH = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    net.load_state_dict(torch.load(f'./saved_models/keypoints_model_{EPOCH}.pt', weights_only=True,map_location=device))
    net.eval()

    for i, sample in enumerate(test_loader):
        image_tensor = sample.type(torch.FloatTensor).to(device)
        outputs = net(image_tensor)
        keypoints = outputs.view(outputs.size()[0], 68, -1)

        print(f"Image tensor size: {image_tensor.data.size()}")
        print(f"Keypoints size: {keypoints.data.size()}")

        for j in range(len(sample)):  # Use len(sample) to handle the last batch correctly
            # Un-transform the predicted key_pts data
            predicted_key_pts = keypoints[j].data
            predicted_key_pts = predicted_key_pts.cpu().numpy()
            # Undo normalization of keypoints  
            predicted_key_pts = predicted_key_pts * 50.0 + 100
            
            image = image_tensor[j].cpu().numpy()  # Convert to numpy array from a Tensor
            image = np.transpose(image, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
            
            plt.imshow(np.squeeze(image), cmap="gray")
            plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
            plt.savefig(f'./faces_{i}_{j}.png')