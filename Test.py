import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm


# Imports here
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import cv2
import os
from tqdm import tqdm
from PIL import Image

# pytorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm


# Imports here
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import cv2
import os
from tqdm import tqdm
from PIL import Image

# pytorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision


# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

# Set model to evaluation mode
transforms_for_function = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
    ]
)

def get_vector(image_path):
  with torch.no_grad():
    image = Image.open(image_path)
    t_img = transforms_for_function(image)
    t_img = t_img.to(device)
    return model(t_img.unsqueeze(0))[0]

def load_images_from_folder(folder):
    images = []
    for filename in tqdm(os.listdir(folder)):
        vector = get_vector(folder + "/" + filename)
        to_append = (vector , filename)
        images.append(to_append)

    return images
# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

# Set model to evaluation mode
transforms_for_function = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
    ]
)

def get_vector(image_path):
  with torch.no_grad():
    image = Image.open(image_path)
    t_img = transforms_for_function(image)
    t_img = t_img.to(device)
    return model(t_img.unsqueeze(0))[0]

def load_images_from_folder(folder):
    images = []
    for filename in tqdm(os.listdir(folder)):
        vector = get_vector(folder + "/" + filename)
        to_append = (vector , filename)
        images.append(to_append)

    return images





FILE = "*******"

model = torch.load(FILE)
model.eval()

torch.cuda.empty_cache()
torch.cuda.max_memory_allocated(device=None)

if torch.cuda.is_available():
  print("GPU is selected, you are good to go")
else:
  print("Go to setting and select GPU as runtime")
torch.cuda.empty_cache()
torch.cuda.max_memory_allocated(device=None)

if torch.cuda.is_available():
  print("GPU is selected, you are good to go")
else:
  print("Go to setting and select GPU as runtime")





general_directory = "*******"
query_directory = general_directory + "/query"  
gallery_directory1 = general_directory + "/gallery"

#LOAD MODEL
MODEL_PATH = "*******"

# we just remove the classifier
model = torch.load(MODEL_PATH)
device = torch.device("cuda:0")
model.classifier = torch.nn.Identity()
# set model to eval mode
model.eval()
MODEL_PATH = "/content/drive/MyDrive/COMPETITION/efficientnet_for_similarity/PATHS/B0_MODELLO.pth"

# we just remove the classifier
model = torch.load(MODEL_PATH)
device = torch.device("cuda:0")
model.classifier = torch.nn.Identity()
# set model to eval mode
model.eval()

query_list = load_images_from_folder(query_directory)query_list = load_images_from_folder(query_directory)
gallery_list1 = load_images_from_folder(gallery_directory1)gallery_list1 = load_images_from_folder(gallery_directory1)

to_sub = {}

cos =  nn.CosineSimilarity(dim=-1)

for query_vector , query_name in tqdm(query_list):
  my_list = []
  for gallery_vector , gallery_name in gallery_list1:
    dist = cos(query_vector, gallery_vector)
    my_list.append( (dist.item(), gallery_name))
  my_list.sort(key=lambda x:x[0], reverse = True)
  to_sub[query_name] = [el[1] for el in my_list ][:10]to_sub = {}


