##pysolar:https://gis.stackexchange.com/questions/229293/shadow-impact-of-building-on-certain-area
from pysolar import solar
from datetime import datetime
from math import sin, cos, radians
import pandas as pd
import numpy as np


def sun_location(altitude, azimuth, from_point, dist_to_sun=100):
    """given an altitude and azimuth to the sun, an assumed
    distance to the sun of 100, and a point of origin 
    find the XYZ location of the sun
    Note: the reference system from pysolar is:
    http://pysolar.readthedocs.io/en/latest/index.html#location-calculation
    The azimuth to the sun is realtive to South, so subtract 270
    to get Easterly-based directions.
    """

    x_from, y_from, z_from = from_point
    # 
    x_sun = dist_to_sun * sin(radians(azimuth-270))
    y_sun = dist_to_sun * sin(radians(altitude))
    z_sun = dist_to_sun * cos(radians(azimuth-270))

    return x_sun, y_sun, z_sun

from datetime import timezone #to specify timezone



if __name__ == "__main__":
    arg = sys.argv[:]
    if len(arg) < 8:
        print('Not enough arguments!\n')
        print('python azimuth_estimation.py long lat date_start date_end spatial_resolution max_elevation path_to_results')
        exit(0)

    long = float(arg[1])
    lat = float(arg[2])
    date_start = str(arg[3])
    date_end = str(arg[4])
    spatial_resolution = float(arg[5])
    max_elevation = float(arg[6])
    path_to_results = str(arg[7])


    # calculate the sun's position in the sky on a given day
    drange = pd.date_range(start=date_start,end=date_end, freq="S")
    pi=22/7
    az_dict = {}
    for date in drange:
        d = datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S')
        d = d.replace(tzinfo=timezone.utc)
        altitude = solar.get_altitude(lat, long, d)
        azimuth = solar.get_azimuth(lat, long, d)
        degree = float(azimuth)#152.71)#input("Input degrees: "))
        radian = degree*(pi/180)
        pix_to_m = np.abs(np.tan(radian))*spatial_resolution*246
        

        if pix_to_m<18:
            az_dict[azimuth]=date

    df_az = pd.DataFrame(az_dict.items(), columns=['azimuth', 'date'])
    df_az.head()
    df_az.to_csv(path_to_results, index=False, header=True)

#python azimuth_estimation.py -3.9837119579315186 5.301122665405273 "2008-01-01 08:00:00" "2008-01-01 18:00:00" 2.5 az.csv