import rasterio as rio
from rasterio.transform import *
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import mask
import sys

# dst_crs = 'EPSG:4326'
def reproj_tiff(in_path,out_path):
	with rio.open(in_path) as src:
	    transform, width, height = calculate_default_transform(
	        src.crs, src.crs, src.width, src.height, *src.bounds)
	    kwargs = src.meta.copy()
	    kwargs.update({'crs': src.crs,'transform': transform,'width': width,'height': height})
	    # Use rasterio package as rio to write out the new projected raster
		# Code uses loop to account for multi-band rasters
	    with rio.open(out_path, 'w', **kwargs) as dst:
		    for i in range(1, src.count + 1):
		        reproject(
		        source=rio.band(src, i),
		        destination=rio.band(dst, i),
		        src_transform=src.transform,
		        src_crs=src.crs,
		        dst_transform=transform,
		        dst_crs=dst_crs,
		        resampling=Resampling.nearest)   


if __name__ == "__main__":
    arg = sys.argv[:]
    if len(arg) < 3:
        print('Not enough arguments!\n')
        print('python reproject_raster.py tif_to_tiff path_to_new_tiff CRS_dst')
        exit(0)
    dst_crs = str(arg[3])
	reproj_tiff(in_path=str(arg[1]),out_path=str(arg[2]))	



#Exemple le Plateau:
# python reproject_raster.py  r"F:\intership_files\abidjan quicbird\abidjancom.tif" r"F:\intership_files\abidjan quicbird\reprojabidjancom.tif" 'EPSG:4326'


#De façon équivalent avec gdal
# !gdalwarp -co compress=JPEG -co PHOTOMETRIC=YCBCR -co TILED=YES -co "BLOCKXSIZE=256" -co "BLOCKYSIZE=256" -s_srs crs_source -t_srs crs_dst 'src_tif' 'dst_tif'


