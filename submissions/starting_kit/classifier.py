#%%file submissions/starting_kit/batch_classifier.py
import numpy as np
from skimage import transform
import torch.nn as nn
import torch.optim as optim
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import time
import itertools
from sklearn.model_selection import train_test_split


class Dataset(th.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def createLossAndOptimizer(net, learning_rate=0.001):
    # it combines softmax with negative log likelihood loss
    criterion = nn.CrossEntropyLoss()  
    #optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    return criterion, optimizer

def get_loader(x,y,batch_size,num_workers=1):
    dataset = Dataset(x, y)
    return th.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,shuffle=True)


def train(net,x,y,val_x,val_y, batch_size=32, n_epochs=10, learning_rate=0.001):
    
    """
    Train a neural network and print statistics of the training
    
    :param  net: (PyTorch Neural Network)
    :param batch_size: (int)
    :param n_epochs: (int)  Number of iterations on the training set
    :param learning_rate: (float) learning rate used by the optimizer
    """
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("n_epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    train_loader = get_loader(x,y, batch_size)
    val_loader = get_loader(val_x,val_y, batch_size)
    n_minibatches = len(train_loader)

    criterion, optimizer = createLossAndOptimizer(net, learning_rate)
    # Init variables used for plotting the loss
    train_history = []
    val_history = []

    training_start_time = time.time()
    best_error = np.inf
    best_model_path = "best_model.pth"
    
    # Move model to gpu if possible
    net = net.to(device)

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        print_every = n_minibatches // 10
        start_time = time.time()
        total_train_loss = 0
        
        for i, (inputs, labels) in enumerate(train_loader):

            # Move tensors to correct device
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total_train_loss += loss.item()

            # print every 10th of epoch
            if (i + 1) % (print_every + 1) == 0:    
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                      epoch + 1, int(100 * (i + 1) / n_minibatches), running_loss / print_every,
                      time.time() - start_time))
                running_loss = 0.0
                start_time = time.time()

        train_history.append(total_train_loss / len(train_loader))

        total_val_loss = 0
        # Do a pass on the validation set
        # We don't need to compute gradient,
        # we save memory and computation using th.no_grad()
        with th.no_grad():
            for inputs, labels in val_loader:
                # Move tensors to correct device
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass
                predictions = net(inputs)
                val_loss = criterion(predictions, labels)
                total_val_loss += val_loss.item()
            
        val_history.append(total_val_loss / len(val_loader))
        # Save model that performs best on validation set
        if total_val_loss < best_error:
            best_error = total_val_loss
            th.save(net.state_dict(), best_model_path)

        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))

    print("Training Finished, took {:.2f}s".format(time.time() - training_start_time))
    
    # Load best model
    net.load_state_dict(th.load(best_model_path))
    
    return train_history, val_history

class BaselineConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(BaselineConvolutionalNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # cf comments in forward() to have step by step comments
        # on the shape (how we pass from a 3x150x150 input image to a 16x75x75 volume)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # cf comments in forward() to have step by step comments
        # on the shape (how we pass from a 16x75x75 input image to a 32x36x36 volume)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # cf comments in forward() to have step by step comments
        # on the shape (how we pass from a 32x36x36 input image to a 32x18x18 volume)
        
        self.flatten = 32 * 18 * 18
        
        self.fc1 = nn.Linear(self.flatten, 64) 
        self.fc2 = nn.Linear(64, 6)

    def forward(self, x):
        """
        Forward pass,
        x shape is (batch_size, 3, 32, 32)
        (color channel first)
        in the comments, we omit the batch_size in the shape
        """
        # shape : 3x150x150 -> 16x150x150
        x = F.relu(self.conv1(x))
        # 16x150x150 -> 16x75x75
        x = self.pool1(x)
        
        # shape : 16x75x75 -> 32x75x75
        x = F.relu(self.conv2(x))
        # 32x75x75 -> 32x38x38
        x = self.pool2(x)
        
        # shape : 32x38x38 -> 32x38x38
        x = F.relu(self.conv3(x))
        # 32x38x38 -> 32x19x19
        x = self.pool3(x)
        
        # 32x19x19 
        x = x.view(-1, self.flatten)
        # 32x19x19  -> 64
        x = F.relu(self.fc1(x))
        # 64 -> 6
        # The softmax non-linearity is applied later (cf createLossAndOptimizer() fn)
        x = self.fc2(x)
        return x



class Classifier(object):
    """
    gen_train, gen_valid... shouldn't be modified
    You can change the rest like criterion, optimizer or even the function _build_model if you want to
    
    Accuracy on train and valid during the epochs are on data that are part of the train data for the
    RAMP challenge, so you shouldn't expect to see the same values.
    """
    def __init__(self):
        self.net = BaselineConvolutionalNetwork()
        self.epochs = 1
        self.batch_size = 32

    def fit(self, train_x, train_y):
        X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
        
        train_history, val_history = train(self.net, X_train, y_train,X_val, y_val, self.batch_size, self.epochs)


    def predict_proba(self, X):
        test_loader = th.utils.data.DataLoader(X,batch_size=4,num_workers=1,shuffle=False)

        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.net = self.net.to(device)
        res = []
        for images in test_loader:
            images = images.to(device)
            outputs = self.net(images)
            _, predicted = th.max(outputs, 1)
            res += list(predicted.numpy())
        results = np.zeros((len(X), 6))
        for i in range(len(res)):
            results[i, res[i]] = 1
        return results

