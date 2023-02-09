import torch
import numpy as np 
import torch.optim as optim 
from torch import nn 
import torch.nn.functional as F 
from constants import image_width, image_height
from helpers import accuracy 
import os 

DEBUG: bool = bool(os.environ.get('DEBUG')) or False
print(f'{DEBUG=}')

batch_size = 64

class CNN_MNet(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.image_width, self.image_height = 28, 28 
        self.input_channels, self.output_channels = 1, 1
        self.conv = nn.Conv2d(in_channels=self.input_channels, 
                                out_channels=self.output_channels, 
                                kernel_size=2, 
                                stride=1, 
                                padding=0, 
                                padding_mode='zeros', 
                                bias=True)
        self.output_height = int(np.floor((self.image_height + 2 * 0 - 1 * (2 - 1) - 1 ) / ( 1) + 1))
        self.output_width = int(np.floor((self.image_width + 2 * 0 - 1 * (2 - 1) - 1 ) / ( 1) + 1))

        print(f'cnn output shape: {self.output_width}, {self.output_height}')
        self.l1 = nn.Linear(self.output_channels*self.output_height*self.output_width, 64)
        self.l2 = nn.Linear(64, 10)

    def forward(self, x): 
        if DEBUG: 
            print('before cnn', x.shape)
        x = F.relu(self.conv(x))
        if DEBUG:
            print('after cnn', x.shape)

        x = x.reshape((x.shape[0], 
                self.output_channels*self.output_height*self.output_width))
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x

train_dataset_images = torch.tensor(np.load('./data/train-images-idx3-ubyte.npy'), dtype=torch.float32)
train_ds = train_dataset_images.unsqueeze(1) # add the number of channels (i.e. rgb) since it is greyscale there is only 1 channel
print(f'training dataset: {train_ds.shape}')
train_dataset_labels = torch.tensor(np.load('./data/train-labels-idx1-ubyte.npy'), dtype=torch.long)
print('Read Training Dataset and Labels')
model = CNN_MNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)# SGD(model.parameters(), lr=0.001, momentum=0)
criterion = nn.CrossEntropyLoss()
epochs = 1
from tqdm import trange 
for epoch in range(epochs):
    print(f'Epoch: {epoch+1}')
    running_loss = 0.0 
    for i in (t := trange(train_dataset_images.shape[0]-batch_size)):
        optimizer.zero_grad() 
        imgs = train_dataset_images[i:i+batch_size, :, :].unsqueeze(1)
        #print(f'batch images shape:', imgs.shape)
        imgs_pred = model(imgs)
        imgs_labels = train_dataset_labels[i:i+batch_size]
        loss = criterion(imgs_pred, imgs_labels)
        loss.backward()
        optimizer.step() 
        t.set_description(f'Loss: {loss.item():.4f}')
        
        running_loss += loss.item() 
    running_loss = running_loss / len(range(train_dataset_images.shape[0]-batch_size))
    y_pred = model(train_ds).argmax(dim=1) 
    print(f'Finished Epoch {epoch+1} {running_loss}, Training Accuracy: \
{accuracy(y_pred, train_dataset_labels):.2f}')
print('Finished Training')

y_pred = model(train_ds).argmax(dim=1) 
print(f'Training Accuracy:{accuracy(y_pred, train_dataset_labels):.2f}')

test_dataset_images = torch.tensor(np.load('./data/t10k-images-idx3-ubyte.npy'), dtype=torch.float32)
test_dataset_images = test_dataset_images.unsqueeze(1)  
print(f'Loaded Test Image Dataset: {test_dataset_images.shape}')
test_dataset_labels = torch.tensor(np.load('./data/t10k-labels-idx1-ubyte.npy'), dtype=torch.long)
test_pred = model(test_dataset_images).argmax(dim=1) 
print(f'Testing Accuracy:{accuracy(test_pred, test_dataset_labels):.2f}')



