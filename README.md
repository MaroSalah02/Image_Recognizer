# Image Classifier Project

This project involves training and using a deep learning model to classify images. The classifier is built using PyTorch, and the project includes scripts for training the model (`train.py`) and predicting the class of an image (`predict.py`).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Predicting Image Class](#predicting-image-class)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites

- Python 3.x
- PyTorch
- torchvision
- numpy
- PIL
- argparse

### Clone the Repository

~~~
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
~~~

### Install Dependencies


~~~
pip install torch torchvision numpy pillow argparse`
~~~
Usage
-----

### Training the Model

To train the model, use the `train.py` script. You need to specify the path to the data directory and other optional parameters like architecture, epochs, learning rate, etc.

#### Example Command


~~~
python train.py path/to/data --arch vgg16 --epochs 10 --learning_rate 0.001 --hidden_units 512 256 --gpu --save_dir path/to/save/checkpoint
~~~

#### Arguments

-   `data_dir` (str): Path to the folder of images.
-   `--save_dir` (str, default='.'): Directory to save the checkpoint.
-   `--arch` (str, default='vgg16'): Model architecture (e.g., `vgg16`, `resnet50`, `densenet121`).
-   `--gpu` (bool, default=False): Use GPU if available.
-   `--hidden_units` (list of int, default=[606]): Hidden units for each layer.
-   `--epochs` (int, default=7): Number of epochs.
-   `--learning_rate` (float, default=0.003): Learning rate.

### Predicting Image Class

To predict the class of an image, use the `predict.py` script. You need to specify the path to the image and the checkpoint file.

#### Example Command

~~~

python predict.py path/to/image path/to/checkpoint --top_k 3 --category_names path/to/category_names.json --gpu
~~~
#### Arguments

-   `image` (str): Path to an image.
-   `checkpoint` (str): Path to the saved trained model.
-   `--gpu` (bool, default=False): Use GPU if available.
-   `--top_k` (int, default=1): Return top K most likely classes.
-   `--category_names` (str): Path to mapping of categories to real names (JSON file).

### Example

To train a model using the VGG16 architecture on a dataset in the `data` folder and save the checkpoint in the `checkpoints` folder
python train.py data --arch vgg16 --epochs 10 --learning_rate 0.001 --hidden_units 512 256 --gpu --save_dir checkpoints

To predict the class of an image using the trained model:

~~~
python predict.py data/test/image_001.jpg checkpoints/checkpoint.pth.tar --top_k 3 --category_names cat_to_name.json --gpu`
~~~
