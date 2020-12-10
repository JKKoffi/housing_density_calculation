"""Merge files into one"""
from rasterio.merge import merge
from glob import glob
import sys

def mergeImageIntoCells(dirpath ,out_fp ):

    #src:https://automating-gis-processes.github.io/CSC18/lessons/L6/raster-mosaic.html


    # Make a search criteria to select the DEM files
    search_criteria = "*.tif"

    q = os.path.join(dirpath, search_criteria)

    img_files = glob(q)


    img_mosaic = []

    for fp in img_files:
        src = rio.open(fp)
        img_mosaic.append(src)
    # Merge function returns a single mosaic array and the transformation info
    mosaic, out_trans = merge(img_mosaic)
    # Copy the metadata
    out_meta = src.meta.copy() 

    out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans,
                         "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
                         }
                        )

    # Write the mosaic raster to disk
    with rio.open(out_fp, "w", **out_meta) as dest:
        dest.write(mosaic)



if __name__ == "__main__":
    arg = sys.argv[:]
    if len(arg) < 3:
        print('Not enough arguments!\n')
        print('python merge_raster.py dir_of_tiles image_path ')
        exit(0)
    dst_crs = str(arg[2])
    mergeImageIntoCells(str(arg[1]),str(arg[2]))
    



#Exemple le Plateau:
# python merge_raster.py  r"F:\intership_files\abidjan quicbird\abidjancom.tif" r"F:\intership_files\abidjan quicbird\reprojabidjancom.tif" 'EPSG:4326'


#test
# python merge_raster.py r"F:\intership_files\tiles\Le_Plateau\Le_Plateau\img" r"F:\intership_files\tiles\Le_Plateau\Le_Plateau\img.tif"
