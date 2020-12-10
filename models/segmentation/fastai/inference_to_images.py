# !curl -LO https://course.fast.ai/setup/colab | bash

# #Installing libraries
# !pip install git+git://github.com/toblerity/shapely.git@master
# !add-apt-repository ppa:ubuntugis/ubuntugis-unstable -y
# !apt-get update
# !apt-get install python-numpy gdal-bin libgdal-dev python3-rtree

# !pip install rasterio
# !pip install geopandas
# !pip install descartes
# !pip install solaris
# !pip install rio-tiler


from fastai.vision import *
from fastai.callbacks import *

from fastai.utils.collect_env import *
# show_install(True)

#perform imports here
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import numpy as np
import shutil
import time
from PIL import Image
import cv2

# import solaris as sol
from solaris.data import data_dir
import os
import shutil
import numpy as np
from PIL import Image
import skimage
import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib.pyplot import *
from shapely.ops import cascaded_union


import skimage.io 
import time

from fastai.vision import *
import torchvision.transforms as T


import rasterio as rio
import solaris as sol

import os

from rasterio.plot import *
import rasterio as rio

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print("Done")


def get_pred(learner, tile):
#     pdb.set_trace()
    #i am converting PIL image to tensor, to give tensor to my model for inference
    
    # img_pil = PIL.Image.open(url)
    img_tensor = T.ToTensor()(tile)
    t_img = Image(img_tensor)
    # inference_learner.predict(img_fastai)

    # t_img =Image(pil2tensor(img_tensor[:,:,:3],np.float32).div_(255))#t_img = Image(pil2tensor(tile[:,:,:3],np.float32).div_(255))
    outputs = learner.predict(t_img)
    im = image2np(outputs[2].sigmoid())
    im = (im*255).astype('uint8')
    return im

""" Inference functions definition """
class SegLabelListCustom(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True, convert_mode='RGB')
    
class SegItemListCustom(SegmentationItemList):
    _label_cls = SegLabelListCustom

def dice_loss(input, target):
#     pdb.set_trace()
    smooth = 1.
    input = torch.sigmoid(input)
    iflat = input.contiguous().view(-1).float()
    tflat = target.contiguous().view(-1).float()
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / ((iflat + tflat).sum() +smooth))

# adapted from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean': return F_loss.mean()
        elif self.reduction == 'sum': return F_loss.sum()
        else: return F_loss

class DiceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, input, target):
        loss = dice_loss(input, target)
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss

class MultiChComboLoss(nn.Module):
    def __init__(self, reduction='mean', loss_funcs=[FocalLoss(),DiceLoss()], loss_wts = [1,1], ch_wts=[1,1,1]):
        super().__init__()
        self.reduction = reduction
        self.ch_wts = ch_wts
        self.loss_wts = loss_wts
        self.loss_funcs = loss_funcs 
        
    def forward(self, output, target):
#         pdb.set_trace()
        for loss_func in self.loss_funcs: loss_func.reduction = self.reduction # need to change reduction on fwd pass for loss calc in learn.get_preds(with_loss=True)
        loss = 0
        channels = output.shape[1]
        assert len(self.ch_wts) == channels
        assert len(self.loss_wts) == len(self.loss_funcs)
        for ch_wt,c in zip(self.ch_wts,range(channels)):
            ch_loss=0
            for loss_wt, loss_func in zip(self.loss_wts,self.loss_funcs): 
                ch_loss+=loss_wt*loss_func(output[:,c,None], target[:,c,None])
            loss+=ch_wt*(ch_loss)
        return loss/sum(self.ch_wts)

def acc_thresh_multich(input:Tensor, target:Tensor, thresh:float=0.5, sigmoid:bool=True, one_ch:int=None)->Rank0Tensor:
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    
#     pdb.set_trace()
    if sigmoid: input = input.sigmoid()
    n = input.shape[0]
    
    if one_ch is not None:
        input = input[:,one_ch,None]
        target = target[:,one_ch,None]
    
    input = input.view(n,-1)
    target = target.view(n,-1)
    return ((input>thresh)==target.byte()).float().mean()

def dice_multich(input:Tensor, targs:Tensor, iou:bool=False, one_ch:int=None)->Rank0Tensor:
    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."
#     pdb.set_trace()
    n = targs.shape[0]
    input = input.sigmoid()
    
    if one_ch is not None:
        input = input[:,one_ch,None]
        targs = targs[:,one_ch,None]
    
    input = (input>0.5).view(n,-1).float()
    targs = targs.view(n,-1).float()

    intersect = (input * targs).sum().float()
    union = (input+targs).sum().float()
    if not iou: return (2. * intersect / union if union > 0 else union.new([1.]).squeeze())
    else: return intersect / (union-intersect+1.0)

#load model    
inference_learner = load_learner(file="BinarySegm-focaldice-stage1-best.pkl",path="models/")

if __name__ == "__main__":
  arg = sys.argv[:]
  if len(arg) < 3:
      print('Not enough arguments!\n')
      print('python inference_to_images.py path_to_images path_to_results ')
      exit(0)

  dir_path = str(arg[1])
  dst_dir = str(arg[2])
  create_dir(dst_dir)
  
  imagepaths = []
    # for dirpath, dirnames, filenames in os.walk(r"F:\intership_files\abidjan quicbird\BINGERVILLE"):
  for dirpath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith('.tif'):
            # print(filename)
                imagepaths.append(os.path.join(dirpath, filename))
              
  N = len(imagepaths)
  j = 0
  for file in imagepaths:
    test_tile = skimage.io.imread(file)
    result = get_pred(inference_learner, test_tile)
    
    src = rio.open(file)
    
    fname = file.split('/')[-1]
    out_path = os.path.join(dst_dir,fname)
    
    result = reshape_as_raster(result)
    metadata = src.meta
    with rasterio.open(out_path, 'w', **metadata) as dst:
          dst.write(result) 
    
    j +=1
    prog = ((j)/len(imagepaths)) * 100
    print('\rCompleted: {:.2f}%'.format(prog),end=' ')

#python  inference_to_images.py /images /results 