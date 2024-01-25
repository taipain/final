import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path
from natsort import natsorted as sort
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-inp",type=str,default="./dataset/test")
args= parser.parse_args()

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(256*256*3, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
            nn.ReLU())

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def transforms_image():
    img_trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    return img_trans

model = MLP()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

print(args.inp)

img_tf = transforms_image()

for i in sort(Path(args.inp).iterdir()):
    image = Image.open(str(i)).convert('RGB')
    image = img_tf(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output).item()
        
    if predicted_class == 0:
        name = "川上洋平"
    elif predicted_class ==1:
        name = "川谷絵音"
    else:
        name = "米津玄師"
    print(f'{str(i)} is: {name}')
