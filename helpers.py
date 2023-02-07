import torch 
def accuracy(y_pred, labels):
    return (torch.sum(y_pred == labels)) / labels.shape[0]


