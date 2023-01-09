import sys
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torchvision import transforms as T
import cv2
import albumentations as A

from PIL import Image
import segmentation_models_pytorch as smp

import json
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
f = open('Config.json')
Config = json.load(f)

job = sys.argv

def detect(args):
    model = torch.load(args.weight, map_location=device)
    model.eval()
    img = cv2.imread(args.input, cv2.COLOR_BGR2RGB)
    resize_back = A.Resize(img.shape[0], img.shape[1], interpolation=cv2.INTER_NEAREST)
    resize = A.Resize(1024, 768, interpolation=cv2.INTER_NEAREST)
    t = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    aug = resize(image=img)
    img = Image.fromarray(aug['image'])
    img = t(img)
    img = img.to(device)
    
    with torch.no_grad():
        img = img.unsqueeze(0)
        output = model(img)
        output = torch.argmax(output, dim=1).cpu().squeeze(0).numpy() * 255
        #output = resize_back(image=output)['image']
    
    fig, ax = plt.subplots(1,2, figsize=(14, 10))
    img = cv2.imread(args.input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax[0].imshow(img)
    ax[0].set_title("Input")
    
    ax[1].imshow(output)
    ax[1].set_title("Output")
    
    fig.savefig("output.jpg")
    cv2.imwrite("mask.jpg", output)
    
    #cv2.imwrite("output.jpg", output)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default="", type=str, help="model pretrained weight")
    parser.add_argument("--input", default="", type=str, help='input image')
    args = parser.parse_args()
    detect(args)
      