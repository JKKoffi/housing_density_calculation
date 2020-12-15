# housing_density_calculation

### About 

The aim of this project is to estimate the number of household by using satellite image like raster file.
We used for this project a raster image of Abidjan city, with 2.5 metter of spatial resolution.

### Poject env setup and activation

```{bash}
git clone https://github.com/cosmiq/solaris.git
```

* for GPU:

```{bash}
conda env create -f environment-gpu.yml
```

* for CPU:

```{bash}
conda env create -f environment-gpu.yml
```

To activate the environment do:
```{bash}
conda activate solaris
```

### Running example script
T run any script of this project, move to ve foder containing the script script_name.py and do:
```
python script_name.py input_arguments
```
If you're using any jupyter notebook, do:

```
cd folder_name_of_the_script
%run -i script_name.py input_arguments
```


### Project description:

|- `download_satellite_images` will alow you to download image from Bing Maps and google Maps
|
|- `Image-processing` contain main operation on images 
|
|- `POI_crawler` just to scrape POI (Point of interest) from google street maps 
|
|- `housing_density_calculation.ipynb` For all example of execution script into this repository
|
|- `Project-detail.pdf` Instructions to follow to exécute all scripts

### Contact
`Email`: koffikouakoujonathan58@gmail.com

### Partnership:
International Data Science Institute - Orange Côte d'Ivoire 

