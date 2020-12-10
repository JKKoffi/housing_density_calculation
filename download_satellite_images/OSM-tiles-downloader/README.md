# OSM satellite maps

**Language:** Python 3.7.x

**Input:** longitude min, latitude min, longitude max, latitude max, zoom level

**Output:** Image bounded by Latitude Longitude rectangle

### Running instructions:

Edit  `tiles_to_tiff.py` and add your mapbox API key


```Batchfile
python tiles_to_tiff.py <lon_min> <lat_min> <lon_max> <lat_max> <zoom-level>
``` 

### Required
- GDAL
- Empty folder `output` and `tmp`

### Références

Website to find bounding box coordinates area are:
* [https://boundingbox.klokantech.com/](https://boundingbox.klokantech.com/) -Tr-s bon site avec des possibilités d'exports
* [http://bboxfinder.com/](http://bboxfinder.com/)
* [http://geojson.io]( http://geojson.io)

website to find gps coordinates are:
* [https://www.latlong.net/](https://www.latlong.net/)
* [https://www.gps-coordinates.net/](https://www.gps-coordinates.net/)
* [https://www.mapcoordinates.net/en](https://www.mapcoordinates.net/en)



