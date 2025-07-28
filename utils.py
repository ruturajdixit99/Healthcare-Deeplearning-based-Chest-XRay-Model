import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models


class_names=['Normal','Pneumonia']

def load_model(model_path):
    model=models.densenet121(pretrained=False)
    num_ftrs=model.classifier.in_features
    model.classifier=nn.Linear(num_ftrs,len(class_names))
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_class(image,model):
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,[0.229]])
    ])
    image=transform(image).unsequeeze(0)
    outputs=model(image)
    _,predicted=torch.max(outputs,1)
    return class_names[predicted.item()]