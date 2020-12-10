"""Split a raster into multiple files and save into folder predict-256"""
#src:https://gis.stackexchange.com/questions/306861/split-geotiff-into-multiple-cells-with-rasterio
from shapely import geometry
from rasterio.mask import mask
import rasterio as rio
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print("Done")


def splitImageIntoCells(fileIn,filename, squareDim):

    img = rio.open(fileIn)
    numberOfCellsWide = img.shape[1] // squareDim
    numberOfCellsHigh = img.shape[0] // squareDim
    x, y = 0, 0
    count = 0
    for hc in range(numberOfCellsHigh):
        y = hc * squareDim
        for wc in range(numberOfCellsWide):
            x = wc * squareDim
            geom = getTileGeom(img.transform, x, y, squareDim)
            getCellFromGeom(img, geom, filename, count)
            count = count + 1

# Generate a bounding box from the pixel-wise coordinates using the original datasets transform property
def getTileGeom(transform, x, y, squareDim):
    corner1 = (x, y) * transform
    corner2 = (x + squareDim, y + squareDim) * transform
    return geometry.box(corner1[0], corner1[1],
                        corner2[0], corner2[1])

# Crop the dataset using the generated box and write it out as a GeoTIFF
def getCellFromGeom(img, geom, filename, count):
    crop, cropTransform = mask(img, [geom], crop=True)
    writeImageAsGeoTIFF(crop,
                        cropTransform,
                        img.meta,
                        img.crs,
                        filename+"_"+str(count))

# Write the passed in dataset as a GeoTIFF
def writeImageAsGeoTIFF(img, transform, metadata, crs, filename):
    metadata.update({"driver":"GTiff",
                     "height":img.shape[1],
                     "width":img.shape[2],
                     "transform": transform,
                     "crs": crs})
    with rio.open(filename+".tif", "w", **metadata) as dest:
        dest.write(img)


if __name__ == "__main__":
    arg = sys.argv[:]
    if len(arg) < 4:
        print('Not enough arguments!\n')
        print('python split_raster.py path_to_tif path_to_save_tiles tile_size_by_256')
        exit(0)
    create_dir(str(arg[2]))
    splitImageIntoCells(fileIn=str(arg[1]),filename=str(arg[2]), squareDim=int(arg[3]))



#Exemple le Plateau:
# python split_raster.py  r"F:\intership_files\abidjan quicbird\COCODY.tif" r"F:\intership_files\abidjan quicbird\Cocody\Cocody" 256*6

