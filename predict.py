# Importing the libraries
import argparse as arg
import torch
from torch import nn ,optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import classes_DL as dl
import json
import time

print("Welcome in Image Classifier (Predict)")
args = dl.get_inputs_predict()
if args.gpu == True:
    device = "cuda"
else:
    device = "cpu"

model, check_point = dl.load_any_model(args.checkpoint,device)
model.to(device)

print("The trained model specification:\n",
      f"Archetict: {check_point['type']}\n",
      f"Epochs: {check_point['epoch']}\n",
      f"Gpu_Enabled: {args.gpu}\n")
image_address = args.image




props, classes = dl.predict(image_address,model,device,args.top_k)


if args.category_names is None:
    print(props)
    print(classes)

else:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    names = np.array([])
    for c in classes:
        names = np.append(names, cat_to_name[str(c)])
    print(props)
    print(classes)
    print(names)
