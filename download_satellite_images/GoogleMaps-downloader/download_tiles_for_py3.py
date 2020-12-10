#!/usr/bin/python

import urllib.request as urllib2
import os, sys
from gmap_utils import *
from datetime import datetime
from PIL import Image

import time
import random


directory = os.path.abspath('./{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
if not os.path.exists(directory):
    os.makedirs(directory)


def download_tiles(zoom, lat_start, lat_stop, lon_start, lon_stop, satellite=True):

    start_x, start_y = latlon2xy(zoom, lat_start, lon_start)
    stop_x, stop_y = latlon2xy(zoom, lat_stop, lon_stop)
    
    print ("x range", start_x, stop_x)
    print ("y range", start_y, stop_y)
    
    user_agent = 'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; de-at) AppleWebKit/533.21.1 (KHTML, like Gecko) Version/5.0.5 Safari/533.21.1'
    headers = { 'User-Agent' : user_agent }
    #Creating temp directory

    TYPE, ext = 's', 'jpg'
    w = (stop_x - start_x) * 256
    h = (stop_y - start_y) * 256
    
    print( "width:", w)
    print ("height:", h)
    
    result = Image.new("RGB", (w, h))
    # result = Image.new("RGBA", (w, h))

    percent = (stop_x-start_x)*(stop_y-start_y)
    n=0

    for x in range(start_x, stop_x):
        for y in range(start_y, stop_y):
            
            url = None
            filename = None
            #1024
            # url = "http://mt0.google.com/vt/lyrs=s&scale=4&s=Galileo&x=%d&s=&y=%d&z=%d" % (x, y, zoom)
            #768
            # url = "http://mt1.google.com/vt/lyrs=s&scale=3&s=Galil&x=%d&s=&y=%d&z=%d" % (x, y, zoom)
            #512
            # url = "http://mt1.google.com/vt/lyrs=s&scale=2&s=Gali&x=%d&s=&y=%d&z=%d" % (x, y, zoom)
            #By default
            url = "https://mt0.google.com/vt/lyrs=s&?x=%d&s=&y=%d&z=%d" % (x, y, zoom)
            # filename = "%d_%d_%d_s.jpg" % (zoom, x, y)
            filename = os.path.join(directory,"%d_%d_%d_s.jpg" % (zoom, x, y))
            
            
            # if satellite:        
            #     url = "http://khm2.google.com/kh/v=708&s=Gal&x=%d&y=%d&z=%d" % (x, y, zoom)
            #     filename = "%d_%d_%d_s.jpg" % (zoom, x, y)
            # else:
            #     url = "http://mt1.google.com/vt/lyrs=h@162000000&hl=en&x=%d&s=&y=%d&z=%d" % (x, y, zoom)
            #     filename = "%d_%d_%d_r.png" % (zoom, x, y)    
    
            if not os.path.exists(filename):
                
                bytes = None
                
                try:
                    req = urllib2.Request(url, data=None, headers=headers)
                    response = urllib2.urlopen(req)
                    bytes = response.read()
                except Exception as e:
                    print ("--", filename, "->", e)
                    sys.exit(1)
                
                if bytes.startswith(b"<html>"):
                    print ("-- forbidden", filename)
                    sys.exit(1)
                n+=1
                print (str(n),f"of {percent} -- saving", filename)
                
                f = open(filename,'wb')
                f.write(bytes)
                f.close()
                
                print('download complete')

                time.sleep(1 + random.random())

    for x in range(start_x, stop_x):
        for y in range(start_y, stop_y):
            
            # filename = "%d_%d_%d_%s.%s" % (zoom, x, y, TYPE, ext)
            filename = os.path.join(directory,"%d_%d_%d_%s.%s" % (zoom, x, y, TYPE, ext))
 
            if not os.path.exists(filename):
                print("-- missing", filename)
                continue
                    
            x_paste = (x - start_x) * 256
            y_paste = h - (stop_y - y) * 256
            
            try:
                i = Image.open(filename)
            except Exception as e:
                print("-- %s, removing %s" % (e, filename))
                trash_dst = os.path.expanduser("~/.Trash/%s" % filename)
                os.rename(filename, trash_dst)
                continue
            
            result.paste(i, (x_paste, y_paste))
            
            del i
    # to convert the type into RGB --> jpeg
    # if result.mode in ("RGBA", "P"):
    #     result = result.convert("RGB")

    result.save("map_%s.%s" % (TYPE, ext),optimize=True)
    print('successfully merged')



if __name__ == "__main__":
    arg = sys.argv[:]
    if len(arg) < 6:
        print('Not enough arguments!\n')
        print('python maine.py <top_lat> <top_long> <bot_lat> <bot_long> <zoom-level>')
        exit(0)

    zoom = float(arg[5])#15

    lon_start,lat_stop ,lon_stop,lat_start = float(arg[1]),float(arg[2]),float(arg[3]),float(arg[4])

    if lat_start == lat_stop or lon_start == lon_stop:
        print('Cannot accept equal latitude or longitude pairs.\nTry with a different combination')
        exit(0)

    download_tiles(zoom, lat_start, lat_stop, lon_start, lon_stop, satellite=True)

#Exemple le Plateau:
# python download_tiles_for_py3.py -4.0279855151 5.3319522642 -4.0249599833 5.3350461774 19