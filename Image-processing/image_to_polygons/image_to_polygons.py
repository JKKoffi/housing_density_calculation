# !add-apt-repository ppa:ubuntugis/ubuntugis-unstable -y
# !apt-get update
# !apt-get install python-numpy gdal-bin libgdal-dev python3-rtree

# !pip install rasterio
# !pip install geopandas
# !pip install descartes
# !pip install solaris
# !pip install rio-tiler

import numpy as np
import rasterio 
from rasterio.windows import Window

import cv2 
import geopandas as gpd

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import cascaded_union
from collections import defaultdict

from pathlib import Path

import matplotlib.pyplot as plt
%matplotlib inline

from tqdm import tqdm

def pad_window(window, pad):
    col_off, row_off, width, height = window.flatten()
    return Window(col_off-pad//2, row_off-pad//2,width+pad,height+pad)

# https://gis.stackexchange.com/questions/271733/geopandas-dissolve-overlapping-polygons
# https://nbviewer.jupyter.org/gist/rutgerhofste/6e7c6569616c2550568b9ce9cb4716a3

def explode(gdf):
    """    
    Will explode the geodataframe's muti-part geometries into single 
    geometries. Each row containing a multi-part geometry will be split into
    multiple rows with single geometries, thereby increasing the vertical size
    of the geodataframe. The index of the input geodataframe is no longer
    unique and is replaced with a multi-index. 

    The output geodataframe has an index based on two columns (multi-index) 
    i.e. 'level_0' (index of input geodataframe) and 'level_1' which is a new
    zero-based index for each single part geometry per multi-part geometry
    
    Args:
        gdf (gpd.GeoDataFrame) : input geodataframe with multi-geometries
        
    Returns:
        gdf (gpd.GeoDataFrame) : exploded geodataframe with each single 
                                 geometry as a separate entry in the 
                                 geodataframe. The GeoDataFrame has a multi-
                                 index set to columns level_0 and level_1
        
    """
    gs = gdf.explode()
    gdf2 = gs.reset_index().rename(columns={0: 'geometry'})
    gdf_out = gdf2.merge(gdf.drop('geometry', axis=1), left_on='level_0', right_index=True)
    gdf_out = gdf_out.set_index(['level_0', 'level_1']).set_geometry('geometry')
    gdf_out.crs = gdf.crs
    return gdf_out


if __name__ == "__main__":
    arg = sys.argv[:]
    if len(arg) < 3:
        print('Not enough arguments!\n')
        print('python image_to_polygons.py tif_url geojson_url ')
        exit(0)


	# padded windowed reads with blocks and rasterio: https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html

	raster_COCODY = rasterio.open(str(arg[1]) ,'r')
	pad = 64
	windows_COCODY = []
	for ji, window in raster_COCODY.block_windows(1):
	    assert len(set(raster_COCODY.block_shapes)) == 1
	    window_padded = pad_window(window, pad)
	    windows_COCODY.append((ji, window_padded))



	mask_thres = 0.7
	epsilon = 10
	min_area = 100.
	erode = 5
	dilate = 7

	polys_COCODY = []

	for window in tqdm(windows_COCODY):
	    win_tnfm = rasterio.windows.transform(window[1], raster_COCODY.meta['transform'])
	    win_img = raster_COCODY.read(window=window[1])[0,:,:]

	    mask = win_img > (mask_thres*255)
	    # erode and dilate
	    mask = cv2.erode(mask.astype('uint8'), np.ones((erode,erode),np.uint8), iterations=1)
	    mask = cv2.dilate(mask.astype('uint8'), np.ones((dilate,dilate),np.uint8), iterations=1)

	    # label via connected components
	    _, instances = cv2.connectedComponents(mask.astype('uint8'))

	    # make polys from instances
	    uniques = list(np.unique(instances))
	    for b in uniques[1:]:
	        poly = mask_to_polygons(instances==b,epsilon, min_area)
	        try: 
	            if poly.type == 'MultiPolygon': 
	                geo_coords = poly2coords(poly, win_tnfm)
	                polys_COCODY.append(geo_coords)
	            #else: print('not a MultiPolygon')
	        except Exception as exc: print(f"{exc}: {window}")


	# dedupe windowed polys
	merged_poly_COCODY = gpd.GeoDataFrame()
	merged_poly_COCODY['geometry'] = gpd.GeoSeries(cascaded_union(polys_COCODY))

	gdf_out_cocody = explode(merged_poly_COCODY)
	gdf_out_cocody = gdf_out_cocody.reset_index()

	# fill holes
	gdf_out_cocody.geometry = gdf_out_cocody.geometry.apply(lambda x: Polygon([coords for coords in x.exterior.coords]))

	gdf_out_cocody.drop(columns=['level_0','level_1'], inplace=True)

	# buffer polys
	gdf_out_cocody['geometry_buffered'] = gdf_out_cocody.geometry.buffer(0.000002, cap_style=1, join_style=1)

	gdf_out_cocody['coords_pixel'] = gdf_out_cocody.geometry_buffered.apply(lambda x: Polygon([np.round(coords*~raster_COCODY.meta['transform']) for coords in x.exterior.coords]))
   
	gdf_out_cocody['coords_geo_rounded'] = gdf_out_cocody.geometry_buffered.apply(lambda x: Polygon([np.round(coords,6) for coords in x.exterior.coords]))

	gdf_out_cocody.geometry = gdf_out_cocody.geometry_buffered

	gdf_out_cocody.geometry.to_file(str(arg[2]), driver='GeoJSON')

#python image_to_polygons.py '/content/tiff/inference_cocody_reproj.tif' f'/content/exports/buildigns_COCODY.geojson'


