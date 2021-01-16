#!/usr/bin/env bash

# echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
# wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh
# /bin/bash ~/anaconda.sh -b -p /opt/conda
# rm ~/anaconda.sh
# export PATH=/opt/conda/bin:$PATH
# pip install --upgrade tensorflow-gpu
pip install --upgrade keras
pip install imgaug lightgbm
# conda install opencv tqdm

# pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
pip install torchvision tensorboardXd

#install requirements
# !add-apt-repository ppa:ubuntugis/ubuntugis-unstable -y
# !apt-get update
# !apt-get install python-numpy gdal-bin libgdal-dev python3-rtree

pip install rasterio
pip install geopandas
pip install descartes
pip install solaris
pip install rio-tiler

# pip install git+git://github.com/toblerity/shapely.git@master


pip install supermercado

#src:https://github.com/fastai/fastai1
# pip install fastai==1.0.61
pip install fastai==1.0.55

