# Imports here
import numpy as np
#import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
#import cv2
import os
from tqdm.auto import tqdm
import time

# pytorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import ColorJitter

matplotlib.style.use('ggplot')

FILE = "*******"
FILE2 = "******"
FILE3 = "******"

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
  print("GPU is selected, you are good to go")
else:
  print("Go to setting and select GPU as runtime")

  

def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(FILE2)
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(FILE3)


# Required constants.
from torchvision.transforms.transforms import ColorJitter

ROOT_DIR = "/content/drive/MyDrive/COMPETITION/training_dataset"
train_dir = "/content/drive/MyDrive/COMPETITION/training_dataset/training"
val_dir = "/content/drive/MyDrive/COMPETITION/training_dataset/new_validation"
IMAGE_SIZE = 224 # Image size of resize when applying transforms.
BATCH_SIZE = 50
NUM_WORKERS = 2 # Number of parallel processes for data preparation.

# Training transforms
def get_train_transform(IMAGE_SIZE, pretrained):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=.5, hue=.3),  
        transforms.RandomPerspective(distortion_scale=0.6, p=0.3), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return train_transform

# Validation transforms
def get_valid_transform(IMAGE_SIZE, pretrained):
    valid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return valid_transform

# Image normalization transforms.
def normalize_transform(pretrained = True):
    if pretrained: # Normalization for pre-trained weights.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )

    else: # Normalization when training from scratch.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    return normalize

def get_datasets(pretrained = True):
    """
    Function to prepare the Datasets.

    :param pretrained: Boolean, True or False.

    Returns the training and validation datasets along 
    with the class names.
    """
    dataset_train = datasets.ImageFolder(
        train_dir, 
        transform=(get_train_transform(IMAGE_SIZE, pretrained))
    )
    dataset_valid = datasets.ImageFolder(
        val_dir, 
        transform=(get_valid_transform(IMAGE_SIZE, pretrained))
    )
    

    return dataset_train, dataset_valid, dataset_train.classes

def get_data_loaders(dataset_train, dataset_valid):
    """
    Prepares the training and validation data loaders.

    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.

    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader


def build_model(pretrained=True, fine_tune=True, num_classes=dataset_classes):
    model_name = 'efficientnet-b4'
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    model = models.efficientnet_b4(pretrained=pretrained)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
            
    # Change the final classification head.
    model.classifier[1] = nn.Linear(in_features=1792, out_features=num_classes)
    return model


def train(model, trainloader, optimizer, criterion):

    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# Validation function.
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc



# Load the training and validation datasets.
dataset_train, dataset_valid, dataset_classes = get_datasets(ROOT_DIR)
print(f"[INFO]: Number of training images: {len(dataset_train)}")
print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
print(f"[INFO]: Class names: {dataset_classes}\n")
# Load the training and validation data loaders.
train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 30
pretrained = True
lr = 0.0001

torch.cuda.empty_cache()
torch.cuda.max_memory_allocated(device=None)

model = build_model(
        pretrained=True, 
        fine_tune=True, 
        num_classes= len(dataset_classes)
    ).to(device)

#early stopping
the_last_loss = 100
trigger_times = 0
patience = 3         # early stopping patience; how long to wait after last time validation loss improved

# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")
# Optimizer.
optimizer = optim.Adam(model.parameters(), lr=lr)
# Loss function.
criterion = nn.CrossEntropyLoss()
# Lists to keep track of losses and accuracies.
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []



# Start the training.
# initialize the early_stopping object

for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                criterion)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    print('-'*50)
    time.sleep(5)

    # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
    if valid_epoch_loss > the_last_loss:
      trigger_times +=1 
      print('trigger times:', trigger_times)

      if trigger_times >= patience:
        print('Early stopping!')
        break

    else:
        print('trigger times: 0')
        trigger_times = 0
      
    the_last_loss = valid_epoch_loss


torch.save(model, FILE )
save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained)

model = torch.load(FILE)
model.eval()
