import geopandas as gpd
import os
from rasterio import mask
from shapely.geometry import box
import rasterio as rio
from pyproj import Proj, transform
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as pyplot
from rasterio.plot import *
import cv2
from shapely.geometry import *
from rasterio import mask
from skimage.color import *
import numpy as np
import warnings
import gc
import numpy as np
import pandas as pd
import rasterio as rio

warnings.filterwarnings('ignore')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print("Done")


if __name__ == "__main__":
    arg = sys.argv[:]
    if len(arg) < 4:
        print('Not enough arguments!\n')
        print('python image_clip_gdf.py building_geojson tif_url tiles_dir ')
        exit(0)

    building_cocody_path = str(arg[1])
    tif_url = str(arg[2])
    tiles_dir = str(arg[3])
	
    building_cocody = gpd.read_file(building_cocody_path)
    create_dir(tiles_dir)
    raster_rgb_cocody = rio.open(tif_url)

    condition = ''
    grid_num = '042'
    CLASSIFY = Path(tiles_dir)
    CLASSIFY.mkdir(exist_ok=True)
    (CLASSIFY/f'{condition}_{grid_num}').mkdir(exist_ok=True)

    for i,row in tqdm(building_cocody[(building_cocody['geometry'].type=='Polygon')].iterrows()):
        
        poly = row['geometry'].buffer(0.00001) # padding around detection to crop
    #     print(poly.bounds)

        inProj = Proj(init='epsg:4326') 
        outProj = Proj(init=raster_rgb_cocody.meta['crs']['init']) # convert to cog crs
        
        # convert from geocoords to display window
        minx, miny = transform(inProj,outProj,*poly.bounds[:2])
        maxx, maxy = transform(inProj,outProj,*poly.bounds[2:])
        ul = raster_rgb_cocody.index(minx, miny)
        lr = raster_rgb_cocody.index(maxx, maxy)
        disp_minx, disp_maxx, disp_miny, disp_maxy = lr[0], (max(ul[0],0)+1), max(ul[1],0), (lr[1]+1)

        if disp_maxx-disp_minx <= 150: disp_maxx += 25; disp_minx-=25; 
        if disp_maxy-disp_miny <= 150: disp_maxy += 25; disp_miny-=25;

        # try:
        window = (max(disp_minx,0), disp_maxx), (max(disp_miny,0), disp_maxy)
        data = raster_rgb_cocody.read(window=window)

        # pk += str(row).zfill(5)
        tile_bgr = cv2.cvtColor(np.rollaxis(data,0,3), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{str(CLASSIFY)}/{condition}_{grid_num}/{grid_num}_{i}_{condition}.jpg", tile_bgr)
        # except:
        #   pass




#python image_clip_gdf.py r"\content\exports\buildigns_BINGERVILLE.geojson" r"rgb_BINGERVILLE.tif" r"F:\intership_files\abidjan quicbird\inference\exports\household_estimation\Bingerville_buildings"
