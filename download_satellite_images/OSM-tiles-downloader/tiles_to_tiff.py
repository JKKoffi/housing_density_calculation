import math
import urllib.request
import os
import glob
import subprocess
import shutil
from tile_convert import bbox_to_xyz, tile_edges
from osgeo import gdal
import sys 
import time


def create_dir(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  print("Done")

#updated
def download_tile(x, y, z, tile_server):
    url = tile_server.replace(
        "{x}", str(x)).replace(
        "{y}", str(y)).replace(
        "{z}", str(z))
    path = f'{temp_dir}/{x}_{y}_{z}.png'
    # urllib.request.urlretrieve(url, path)
    try:
        urllib.request.urlretrieve(url, path)
    except urllib.error.HTTPError as exception:
        code =  exception.code
        while (code==404):
           print('wait')
           time.sleep(60)
           try:
               urllib.request.urlretrieve(url, path)
           except urllib.error.HTTPError as exception:
               code = exception.code
    return(path)


def merge_tiles(input_pattern, output_path):
    merge_command = ['gdal_merge.py', '-o', output_path]

    for name in glob.glob(input_pattern):
        merge_command.append(name)

    subprocess.call(merge_command)


def georeference_raster_tile(x, y, z, path):
    bounds = tile_edges(x, y, z)
    filename, extension = os.path.splitext(path)
    gdal.Translate(filename + '.tif',
                   path,
                   outputSRS='EPSG:4326',
                   outputBounds=bounds)





#---------- CONFIGURATION -----------#
tile_server = "https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.png?access_token=pk.eyJ1Ijoiam9uYXRoYW5jaSIsImEiOiJja2JtMXhtOGcwbDBtMzZyNW85bDRoc3NoIn0.WfGAiFLRs8EqoT24h5uVoA"
temp_dir = os.path.join(os.path.dirname('__file__'), 'temp')
output_dir = os.path.join(os.path.dirname('__file__'), 'output')

create_dir(temp_dir)
create_dir(output_dir)
# zoom = 16
# lon_min = 21.49147
# lon_max = 21.5
# lat_min = 65.31016
# lat_max = 65.31688


# lon_min,lat_min,lon_max,lat_max = -4.1589763974,5.2067550578,-3.8281243843,5.4117067371
#-----------------------------------#

arg = sys.argv[:]
if len(arg) < 6:
    print('Not enough arguments!\n')
    print('python ttiles_to_tiff.py <lon_min> <lat_min> <lon_max> <lat_max> <zoom-level>')
    exit(0)

zoom = float(arg[5])#15

lon_min,lat_min,lon_max,lat_max = float(arg[1]),float(arg[2]),float(arg[3]),float(arg[4])

if lat_min == lat_max or lon_min == lon_max:
    print('Cannot accept equal latitude or longitude pairs.\nTry with a different combination')
    exit(0)


x_min, x_max, y_min, y_max = bbox_to_xyz(
    lon_min, lon_max, lat_min, lat_max, zoom)

print(f"Downloading {(x_max - x_min + 1) * (y_max - y_min + 1)} tiles")

for x in range(x_min, x_max + 1):
    for y in range(y_min, y_max + 1):
        print(f"{x},{y}")
        png_path = download_tile(x, y, zoom, tile_server)
        georeference_raster_tile(x, y, zoom, png_path)

print("Download complete")

print("Merging tiles")
merge_tiles(temp_dir + '/*.tif', output_dir + '/merged.tif')
print("Merge complete")

shutil.rmtree(temp_dir)
os.makedirs(temp_dir)

#example to run 
#python tiles_to_tiff.py 5.3319522642 -4.0279855151  5.3350461774 -4.0249599833  16 