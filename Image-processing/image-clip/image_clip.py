

import rasterio as rio
from rasterio import mask
from shapely.geometry import *
from shapely import wkt
import geopandas as gpd
import sys


def clip_tiff(out_path,geo,out_img):

	#reading geometry
	geo = gpd.read_file(geo).to_crs(rio.open(out_path).meta['crs'])
	# geo['geometry'][0] = geo['geometry'].apply(wkt.loads)
	if len(geo.geometry)==1:
		shapes = geo["geometry"][0]
		# shapes = geo

		with rio.open(out_path) as src:
		    out_image, out_transform =mask.mask(src,[ shapes], all_touched=True, crop=True)
		    out_meta = src.meta    

			#to save the result
		out_meta.update({"driver": "GTiff",
		   "height": out_image.shape[1],
		   "width": out_image.shape[2],
		   "transform": out_transform}
		           )
		with rio.open(out_img, "w", **out_meta) as dest:
		    dest.write(out_image)

if __name__ == "__main__":
    arg = sys.argv[:]
    if len(arg) < 4:
        print('Not enough arguments!\n')
        print('python image_clip.py path_to_tif path_to_geojson path_of_new_images')
        exit(0)
    clip_tiff(str(arg[1]),str(arg[2]), str(arg[3]))

