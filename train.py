# Imports here
import argparse
import numpy as np
import torch
import torchvision
from torch import nn, optim
from torchvision import models, datasets, transforms
from PIL import Image
from collections import OrderedDict
from torch.utils.data import DataLoader
import time

# Defining Parser
parser=argparse.ArgumentParser(description='Image Classifier Model Training')

parser.add_argument('--data_dir', help = 'Images directory', type = str, default = './flowers/')
parser.add_argument('--gpu', help= 'GPU', type = str, default = 'gpu')
parser.add_argument('--arch', help = 'Choose model architecture (def = densenet121)', type = str, default = 'densenet121')
parser.add_argument('--learn_rate', help = 'Learning rate (def = 0.001)', type = float, default = 0.001)
parser.add_argument('--hidden_layers', help = 'Hidden layers (def = 420)', type = int, default  = 420)
parser.add_argument('--epochs', help = 'Number of training epochs (def = 3)', type = int, default = 3)

args = parser.parse_args()
data_dir = args.data_dir
gpu = args.gpu
arch = args.arch
learn_rate = args.learn_rate
hidden_layers = args.hidden_layers
epochs = args.epochs

def prep_data(data_dir):
    # Preparation of training, validation and testing data with transforms
    # Defining data sets and loaders

    # Transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(40),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
    
    # Data sets    
    train_data = datasets.ImageFolder(data_dir + '/train', transform = train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform = valid_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform = test_transforms)

    #Data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    
    print("Data preparation finished...")
    return train_data, valid_data, test_data, train_loader, valid_loader, test_loader


#train_data, valid_data, test_data, train_loader, valid_loader, test_loader = prep_data(data_dir)


# Defining function for choosing model architecture
def choose_arch(arch = 'densenet121', hidden_layers = 420):

    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        inputs = 1024
    elif arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        inputs = 25088
    
    # Freeze parameters of pretrained model
    for param in model.parameters():
        param.requires_grad = False
   
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(inputs, hidden_layers)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.4)),
        ('fc2', nn.Linear(hidden_layers, 102)),
        ('output', nn.LogSoftmax(dim=1))]))
        
    model.classifier = classifier
    print("Architecture preparation finished...")
    return model   


# Defining function for model training
def model_train(gpu, model, criterion, optimizer, epochs, train_loader, valid_loader, test_loader):
    
    start = time.time()
    print("Training started")
    steps = 0
    print_every = 5
    
    device=args.gpu
    if device and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    model.to(device)
    
    # Training
    for epoch in range(epochs):
        running_loss = 0
        
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            # Validation
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        valid_loss = criterion(logps, labels)
                    
                        valid_loss += valid_loss.item()
                    
                        # Accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(valid_loader):.3f}"
                    f"\nTime: {round(time.time() - start,2)}")
                running_loss = 0
                model.train()    


# Defining function for testing the model accuracy
def model_test(model, test_loader, gpu):
    
    device=args.gpu
    if device and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    test_loss = 0
    test_accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            test_outputs = model.forward(inputs)
            test_loss = criterion(test_outputs, labels)
                    
            test_loss += test_loss.item()
                    
            # Accuracy
            ps = torch.exp(test_outputs)
            top_p, top_class = ps.topk(1, dim=1)
            test_equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(test_equals.type(torch.FloatTensor)).item()
                    
        print(f"Test accuracy: {100* test_accuracy/len(test_loader):.3f}")
    model.train()



# Preparing data
prep_data(data_dir)
train_data, valid_data, test_data, train_loader, valid_loader, test_loader = prep_data(data_dir)

# Choosing model architecture
model = choose_arch()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = float(args.learn_rate))

# Training a model
model_train(gpu, model, criterion, optimizer, epochs, train_loader, valid_loader, test_loader)

# Testing a model
model_test(model, test_loader, gpu)

# Saving checkpoint
model.class_to_idx = train_data.class_to_idx

checkpoint = {'input_size': 1024,
              'output_size': 102,
              'arch': 'densenet161',
              'classifier' : model.classifier,
              'learn_rate': 0.001,
              'epochs': epochs,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
             }

torch.save(checkpoint, 'model_checkpoint2.pth')