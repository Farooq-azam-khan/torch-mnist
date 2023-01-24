import torch
import numpy as np 
import torch.optim as optim 
from torch import nn 
import torch.nn.functional as F 

image_width = 28 
image_height = 28 

class MNet(nn.Module): 
    def __init__(self): 
        super().__init__()

        self.l1 = nn.Linear(image_height*image_width, 800)
        self.l2 = nn.Linear(800, 10)
        self.output = nn.Softmax(dim=1)

    def forward(self, x): 
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return self.output(x)


def accuracy(y_pred, labels):
    return (torch.sum(y_pred == labels)) / labels.shape[0]

train_dataset_images = torch.tensor(np.load('./data/train-images-idx3-ubyte.npy'), dtype=torch.float32)
train_ds = train_dataset_images.reshape((-1, image_width*image_height))
train_dataset_labels = torch.tensor(np.load('./data/train-labels-idx1-ubyte.npy'), dtype=torch.long)
print('Read Training Dataset and Labels')
model = MNet()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)
criterion = nn.NLLLoss()
batch_size = 32
epochs = 3
from tqdm import trange 
for epoch in range(epochs):
    print(f'Epoch: {epoch+1}')
    running_loss = 0.0 
    for i in trange(train_dataset_images.shape[0]-batch_size):
        optimizer.zero_grad() 
        imgs = train_dataset_images[i:i+batch_size, :, :].reshape((-1, image_width*image_height))
        imgs_pred = model(imgs)
        imgs_labels = train_dataset_labels[i:i+batch_size]
        loss = criterion(imgs_pred, imgs_labels)
        loss.backward()
        optimizer.step() 

        running_loss += loss.item() 
    y_pred = model(train_ds).argmax(dim=1) 
    print(f'Finished Epoch {epoch+1} {running_loss/batch_size}, Training Accuracy: {accuracy(y_pred, train_dataset_labels)}')
print('Finished Training')

y_pred = model(train_ds).argmax(dim=1) 
print(f'Training Accuracy:{accuracy(y_pred, train_dataset_labels):.2f}')

test_dataset_images = torch.tensor(np.load('./data/t10k-images-idx3-ubyte.npy'), dtype=torch.float32)
test_dataset_labels = torch.tensor(np.load('./data/t10k-labels-idx1-ubyte.npy'), dtype=torch.long)
test_ds = test_dataset_images.reshape((-1, image_width*image_height))
test_pred = model(test_ds).argmax(dim=1) 
print(f'Testing Accuracy:{accuracy(test_pred, test_dataset_labels):.2}')




