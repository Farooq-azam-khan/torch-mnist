import torch
import numpy as np 
import torch.optim as optim 
from torch import nn 
import torch.nn.functional as F 
from constants import image_width, image_height
from helpers import accuracy 
import os 

DEBUG = os.environ['DEBUG'] or 0 
class CNN_MNet(nn.Module): 
    def __init__(self): 
        super().__init__()

        self.conv = nn.Conv2d(in_channels=1, 
                                out_channels=1, 
                                kernel_size=1, 
                                stride=1, 
                                padding=0, 
                                padding_mode='zeros', 
                                bias=True)
        self.l1 = nn.Linear((image_height-4)*(image_width-4), 64)
        self.l2 = nn.Linear(64, 10)
        self.opt = nn.Softmax(dim=1) # not working as expected (lower testing accuracy)

    def forward(self, x): 
        print('Running Conv Layer')
        x = self.conv(x)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x

train_dataset_images = torch.tensor(np.load('./data/train-images-idx3-ubyte.npy'), dtype=torch.float32)
train_ds = train_dataset_images.reshape((-1, image_width*image_height))
train_dataset_labels = torch.tensor(np.load('./data/train-labels-idx1-ubyte.npy'), dtype=torch.long)
print('Read Training Dataset and Labels')
model = CNN_MNet()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)
criterion = nn.CrossEntropyLoss()
batch_size = 32
epochs = 1
from tqdm import trange 
for epoch in range(epochs):
    print(f'Epoch: {epoch+1}')
    running_loss = 0.0 
    for i in (t := trange(train_dataset_images.shape[0]-batch_size)):
        optimizer.zero_grad() 
        imgs = train_dataset_images[i:i+batch_size, :, :].unsqueeze(1)#.reshape((-1, image_width*image_height))
        print(imgs.shape)
        imgs_pred = model(imgs)
        imgs_labels = train_dataset_labels[i:i+batch_size]
        loss = criterion(imgs_pred, imgs_labels)
        loss.backward()
        optimizer.step() 
        t.set_description(f'Loss: {loss.item():.2f}')

        running_loss += loss.item() 
    running_loss = running_loss / len(range(train_dataset_images.shape[0]-batch_size))
    y_pred = model(train_ds).argmax(dim=1) 
    print(f'Finished Epoch {epoch+1} {running_loss}, Training Accuracy: {accuracy(y_pred, train_dataset_labels)}')
print('Finished Training')

y_pred = model(train_ds).argmax(dim=1) 
print(f'Training Accuracy:{accuracy(y_pred, train_dataset_labels):.2f}')

test_dataset_images = torch.tensor(np.load('./data/t10k-images-idx3-ubyte.npy'), dtype=torch.float32)
test_dataset_labels = torch.tensor(np.load('./data/t10k-labels-idx1-ubyte.npy'), dtype=torch.long)
test_ds = test_dataset_images.reshape((-1, image_width*image_height))
test_pred = model(test_ds).argmax(dim=1) 
print(f'Testing Accuracy:{accuracy(test_pred, test_dataset_labels):.2}')




