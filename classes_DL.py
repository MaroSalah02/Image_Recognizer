import argparse as arg
import torch
from torch import nn ,optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
import numpy as np
from PIL import Image


#Classifier (Creating a network with relu and sigmoid functions)
class classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, drop=0.2):
        super().__init__()
        self.hidden_units = nn.ModuleList([nn.Linear(input_size,hidden_units[0])])
        layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
        self.hidden_units.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.fc = nn.Linear(hidden_units[-1], output_size)
        self.dropout = nn.Dropout(p = drop)
    def forward(self, x):
        for i in  self.hidden_units:
            x = F.relu(i(x))
            x = self.dropout(x)
        x = F.log_softmax(self.fc(x), dim = 1)
        return x

# Pull the data and apply the transformation needed 
def dataset_transforms(args):
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.Resize(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])


    val_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])


    test_transforms =transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    train_datasets = datasets.ImageFolder(train_dir,transform=train_transforms)
    val_datasets = datasets.ImageFolder(valid_dir,transform=val_transforms)
    test_datasets = datasets.ImageFolder(test_dir,transform=test_transforms)

    trainloaders = torch.utils.data.DataLoader(train_datasets,batch_size=64,shuffle = True)
    valloaders = torch.utils.data.DataLoader(train_datasets,batch_size=64,shuffle = True)
    testloaders = torch.utils.data.DataLoader(test_datasets,batch_size=64,shuffle = True)
    return train_datasets, val_datasets, test_datasets, trainloaders, valloaders ,testloaders

#Get the input from terminal
def get_inputs_train():
    parser = arg.ArgumentParser()
    parser.add_argument('data_dir', type=str, 
                        help='path to folder of images')
    parser.add_argument('--save_dir', type=str,
                        default='.', help='checkpoint saved directory')
    parser.add_argument('--arch', type=str,
                        default='vgg16', help='model archetictor, You can choose (vgg16,resnet15, densenet121) or others available modules that can fit.')
    parser.add_argument('--gpu',action='store_true' ,
                        help='gpu or cpu')
    parser.add_argument('--hidden_units',nargs='+',
                        default=[606] , help='hidden_units for example --hidden_units 400 200 every number is an extra layer')
    parser.add_argument('--epochs',type = int,
                        default=7, help='epoch')
    parser.add_argument("--learning_rate",type = float,
                        default=0.003 , help='learning rate')
    args = parser.parse_args()
    return args

#get inputs from the user to apply custom outputs
def get_inputs_predict():
    parser = arg.ArgumentParser()
    parser.add_argument('image', type=str, 
                        help='path to an image')
    parser.add_argument('checkpoint', type=str,
                        default='.', help='path to saved trained model')
    parser.add_argument('--gpu',action='store_true' ,
                        help='gpu or cpu')

    parser.add_argument('--top_k',type = int,
                        default=1, help='Return top n most likely classes')
    parser.add_argument("--category_names",type = str,
                        help='path to mapping of categories to real names')
    args = parser.parse_args()
    return args


#create a model to be used in training
def create_model(args):
    model = eval("models."+args.arch+"(pretrained=True)")
    if args.arch == "resnet50":
        input_size = 2048
    if args.arch == "vgg16":
        input_size = 25088
    if args.arch == "densenet121":
        input_size = 1024
    
    model.classifier = classifier(input_size,102,list(map(int,args.hidden_units)))
    critaion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr = args.learning_rate )
    return model, critaion, optimizer,input_size


#Create a model for the prediction
def create_model_predict(arch, input_size,output_size,learning_rate):
    model = eval("models."+arch+"(pretrained=True)")
    if arch == "resnet50":
        input_size = 2048
    if arch == "vgg16":
        input_size = 25088
    if arch == "densenet121":
        input_size = 1024
    model.classifier = classifier(input_size,102,list(map(int,args.hidden_units)))
    critaion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr = learning_rate )
    return model, critaion, optimizer


# A function for loading A saved trained network.
def load_any_model(filepath, arch):
    if arch == "coda":
        check_point = torch.load(filepath)
    else:
        check_point = torch.load(filepath,map_location="cpu")
    new_model = eval('models.'+check_point['type']+'(pretrained=True)')
    new_model.class_to_idx = check_point['class_to_idx']
    
    new_model.classifier =  classifier(check_point['input_size'],
                                       check_point['output_size'],
                                       check_point['hidden_layers'])
    new_model.load_state_dict(check_point['state_dict'])
    return new_model,check_point

#Resize then apply the normalization to be used in prediction
def process_image(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = image.resize((256,256))
    image = image.crop((16,16,240,240))
    image = np.array(image)/255
    image = (image - mean) / std
    return image.transpose((2,0,1))


#Predicting the type of flower from the image
def predict(image_path, model,device, topk):
    
    model.eval()
    im = Image.open(image_path)
    im = torch.from_numpy(process_image(im))
    im = torch.unsqueeze(im, 0).float()
    im = im.to(device)
    with torch.no_grad():
        output = model.forward(im)
        
    ex = torch.exp(output)
    top_val, top_ind = ex.topk(topk)
    top_ind = top_ind[0].cpu().numpy()
    classess = np.array([x for x in model.class_to_idx.keys()])[top_ind]
    return top_val[0].cpu().numpy(),classess


