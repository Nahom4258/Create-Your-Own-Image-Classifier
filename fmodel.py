# Imports here
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from collections import OrderedDict
import numpy as np
from PIL import Image
import torchvision.models as models
from torch import nn, optim

def setup_network(architecture='vgg16', hidden_units=4096, lr=0.001, use_gpu=False):
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    if use_gpu:
        model = model.to('cuda')
        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    return model, criterion, optimizer

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    lr = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    architecture = checkpoint['structure']

    model, _ = setup_network(architecture, hidden_units, lr)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def predict(img_path, model, k_number, use_gpu):
    if use_gpu:
        model.to('cuda')
        
    model.eval()
    img = process_image(img_path).numpy()
    img = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
        output = model.forward(img.cuda())
        
    probs = torch.exp(output).data
    
    return probs.topk(k_number)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image)
    
    # Just like the training images
    img_transforms = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    return img_transforms(img_pil)