# Imports here
import argparse
from PIL import Image
import json
from torchvision import datasets, transforms, models
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader


# Defining parser
parser=argparse.ArgumentParser(description='Prediction function')

parser.add_argument('--image', help = 'Images directory', type = str, default = './flowers/test/12/image_03994.jpg') # Default path to colt's foot image
parser.add_argument('--checkpoint', help = 'Path to checkpoint', type = str, default = "./model_checkpoint4.pth")
parser.add_argument('--top_k', help = 'Top k images (def = 5)', type = int, default = 5)
parser.add_argument('--category_names', help = 'Path to category names mapping', type = str, default = 'cat_to_name.json')
parser.add_argument('--gpu', help= 'GPU', type = str, default = 'gpu')

args = parser.parse_args()
image = args.image
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu


# Check if CUDA available
device=args.gpu
if device and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Defining load_checkpoint function
def load_checkpoint(checkpoint):
    
    checkpoint = torch.load(checkpoint)
    
    t_models = {
        'vgg16': models.vgg16(pretrained = True),
         'densenet161': models.densenet161(pretrained = True)
        }
        
    model = t_models.get(checkpoint['arch'],'densenet161')
    
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
        
    return model


def process_image(image):
    
    image = Image.open(args.image)
# Using same transforms as in data preparation in train.py function  
    transformsations = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    processed_image = transformsations(image)
    
    np_image = processed_image.numpy()
    
    return np_image


def predict(image, model, top_k = args.top_k):
    
    image = Image.open(image)
    
    # Processing image
    image = process_image(image)
    image = torch.from_numpy(image).float()   #   Processing image created with mentor's assistance
    image = image.unsqueeze(0)
    image.to('cpu')
    
    # Loading model
    load_checkpoint(checkpoint)
    model.to('cpu')   # Mentor's suggestion
    
    # Processing image through model
    model.eval()
    with torch.no_grad():
        
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        
            log_probs = model.forward(image)
            probs = torch.exp(log_probs)
        
            top_probs, top_classes = probs.topk(top_k)
            top_probs = np.array(top_probs.detach())[0]    #Solution inspired by comments in Udacity Knowledge base
            top_classes = np.array(top_classes.detach())[0]
        
            idx_to_class = {v : k for k, v in model.class_to_idx.items()}    #Solution inspired by comments in Udacity Knowledge base
            top_classes = [idx_to_class[i] for i in top_classes]
            flowers = [cat_to_name[i] for i in top_classes]
        
            return top_probs, top_classes, flowers
    


loaded_model = load_checkpoint(checkpoint)    

top_probs, top_classes, flowers = predict(image, loaded_model, top_k)
top_probability = round(top_probs[0] * 100, 2)

print("This flower is predicted to be {}, with {} % probability".format(flowers[0], top_probability))
print("Other probable flowers are predicted as 1.{}, 2.{}, 3.{}, 4.{}, with probabilities respectively {}%, {}%, {}%, {}%".format(flowers[1], flowers[2], flowers[3], flowers[4], round(top_probs[1] * 100, 2), round(top_probs[2] * 100, 2), round(top_probs[3] * 100, 2), round(top_probs[4] * 100, 2)))



