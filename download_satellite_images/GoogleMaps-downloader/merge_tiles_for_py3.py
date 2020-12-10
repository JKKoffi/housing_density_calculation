#https://github.com/nst/gmap_tiles
from PIL import Image
import sys, os
from gmap_utils import *

def merge_tiles(zoom, lat_start, lat_stop, lon_start, lon_stop, satellite=True):
    
    # TYPE, ext = 'r', 'png'
    # if satellite:
    TYPE, ext = 's', 'jpg'
    
    x_start, y_start = latlon2xy(zoom, lat_start, lon_start)
    x_stop, y_stop = latlon2xy(zoom, lat_stop, lon_stop)
    
    print ("x range", x_start, x_stop)
    print ("y range", y_start, y_stop)
    
    w = (x_stop - x_start) * 256
    h = (y_stop - y_start) * 256
    
    print ("width:", w)
    print ("height:", h)
    
    result = Image.new("RGBA", (w, h))
    
    for x in range(x_start, x_stop):
        for y in range(y_start, y_stop):
            
            filename = "%d_%d_%d_%s.%s" % (zoom, x, y, TYPE, ext)
            
            if not os.path.exists(filename):
                print( "-- missing", filename)
                continue
                    
            x_paste = (x - x_start) * 256
            y_paste = h - (y_stop - y) * 256
            
            try:
                i = Image.open(filename)
            except Exception as e:
                print("-- %s, removing %s" % (e, filename))
                trash_dst = os.path.expanduser("~/.Trash/%s" % filename)
                os.rename(filename, trash_dst)
                continue
  
            result.paste(i, (x_paste, y_paste))
            
            del i
    if result.mode in ("RGBA", "P"):
        result = result.convert("RGB")

    result.save("map_%s.%s" % (TYPE, ext))
    print('successfully merged')


if __name__ == "__main__":
    arg = sys.argv[:]
    if len(arg) < 6:
        print('Not enough arguments!\n')
        print('python merge_tiles _for_py3.py _for_py3.py <top_lat> <top_long> <bot_lat> <bot_long> <zoom-level>')
        exit(0)

    zoom = float(arg[5])#15

    lon_start,lat_stop ,lon_stop,lat_start = float(arg[1]),float(arg[2]),float(arg[3]),float(arg[4])

    if lat_start == lat_stop or lon_start == lon_stop:
        print('Cannot accept equal latitude or longitude pairs.\nTry with a different combination')
        exit(0)

    merge_tiles(zoom, lat_start, lat_stop, lon_start, lon_stop, satellite=True)

#Exemple le Plateau:
# python merge_tiles_for_py3.py  -4.0279855151 5.3319522642 -4.0249599833 5.3350461774 19
