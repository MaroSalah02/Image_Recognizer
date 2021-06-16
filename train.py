#Importing the reqired libraries
import argparse as arg
import torch
from torch import nn ,optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import classes_DL as dl
import time

#Welcoming
print("Welcome in Image Classifier (train)\n")
time.sleep(1)
args = dl.get_inputs_train()

#showing the model specifications

print("Creating model.....")
# importing the data and imply the transformation needed
train_datasets, val_datasets, test_datasets, trainloaders, valloaders ,testloaders = dl.dataset_transforms(args)

#Creating the model
model, critaion, optimizer,input_size = dl.create_model(args)

print("Your model specification:\n",
      f"Archetict: {args.arch}\n",
      f"Epochs: {args.epochs}\n",
      f"Learning_rate: {args.learning_rate}\n",
      f"Hidden_units: {args.hidden_units}\n",
      f"Gpu_Enabled: {args.gpu}\n"
      f"Save_dir: {args.save_dir}\n")
time.sleep(1)
#checking if gpu will be used or not
if args.gpu == True:
    device = "cuda"
    torch.cuda.empty_cache()
else:
    device = "cpu"

model.to(device)

# Training the network 
epochs = args.epochs
steps = 0
i = 0
print_every = 20
steps = 0
running_loss = 0
print("Started the training....")
for epoch in range(epochs):
    for image, label in trainloaders:
        steps += 1
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        logsft = model.forward(image)
        loss = critaion(logsft, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                model.eval()
                for image, label in valloaders:
                    
                    image, label = image.to(device), label.to(device)
                    logsft = model.forward(image)
                    
                    batch_loss = critaion(logsft,label)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logsft)
                    top_p, top_class = ps.topk(1, dim=1)
                    equally = top_class == label.view(*top_class.shape)
                    accuracy += torch.mean(equally.type(torch.FloatTensor)).item()
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"validation loss: {test_loss/len(valloaders):.3f}.. "
                  f"validation accuracy: {accuracy/len(valloaders):.3f}")
            running_loss = 0
            model.train()

#Testing the network            
print("Started the testing....")

test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    model.eval()
    for image, label in testloaders:
        
        image, label = image.to(device), label.to(device)
        logsft = model.forward(image)
        
        batch_loss = critaion(logsft,label)
        test_loss += batch_loss.item()
        
        ps = torch.exp(logsft)
        top_p, top_class = ps.topk(1, dim=1)
        equally = top_class == label.view(*top_class.shape)
        accuracy += torch.mean(equally.type(torch.FloatTensor)).item()
print(f"test loss: {test_loss/len(testloaders):.3f}.. "
      f"test accuracy: {accuracy/len(testloaders):.3f}")
running_loss = 0


#Saving the network with the required data
print("Saving.....")
model.class_to_idx = train_datasets.class_to_idx

checkpoint = {'input_size': input_size,
              'output_size': 102,
              'type':args.arch,
              'epoch':args.epochs,
              'hidden_layers': [i.out_features for i in model.classifier.hidden_units],
              'state_dict': model.state_dict(),
              'optimizer':optimizer.state_dict(),
              'class_to_idx': model.class_to_idx}
file_path = args.save_dir+'/checkpoint.pth.tar'
torch.save(checkpoint, file_path)
print("Done, Your file name and address is : "+ file_path)