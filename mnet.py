import torch
from torch import nn 
import torch.nn.functional as F 
class MNet(nn.Module): 
    def __init__(self): 
        super().__init__()

        self.l1 = nn.Linear(28*28, 200)
        self.l2 = nn.Linear(200, 10)
        self.output = nn.Softmax(dim=1)

    def forward(self, x): 
        return self.output(self.l2(F.relu(self.l1(x))))
import numpy as np 
image_width = 28 
image_height = 28 
train_dataset_images = np.load('./data/train-images-idx3-ubyte.npy')
train_dataset_labels = np.load('./data/train-labels-idx1-ubyte.npy')
img1 = torch.tensor(train_dataset_images[0:10, :, :], dtype=torch.float32).reshape((-1, image_width*image_height))
model = MNet()
img1_pred = model(img1)
print(img1_pred)
print(img1_pred.argmax(dim=1))
print(train_dataset_labels[0])

