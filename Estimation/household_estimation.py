import geopandas as gpd
import os
from rasterio import mask
from shapely.geometry import box
import rasterio as rio
from pyproj import Proj, transform
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as pyplot
from rasterio.plot import *
import cv2
from shapely.geometry import *
from rasterio import mask
from skimage.color import *
import numpy as np
import warnings
import gc
import numpy as np
import pandas as pd
import rasterio
from skimage.color import *
from skimage.exposure import rescale_intensity
from skimage.morphology import disk
from sklearn.cluster import KMeans


warnings.filterwarnings('ignore')





def shadow_detection(image_file, convolve_window_size = 5, num_thresholds = 3, struc_elem_size = 5):
    """
    This function is used to detect shadow - covered areas in an image, as proposed in the paper 
    'Near Real - Time Shadow Detection and Removal in Aerial Motion Imagery Application' by Silva G.F., Carneiro G.B., 
    Doth R., Amaral L.A., de Azevedo D.F.G. (2017)
    
    Inputs:
    - image_file: Path of image to be processed for shadow removal. It is assumed that the first 3 channels are ordered as
                  Red, Green and Blue respectively
    - shadow_mask_file: Path of shadow mask to be saved
    - convolve_window_size: Size of convolutional matrix filter to be used for blurring of specthem ratio image
    - num_thresholds: Number of thresholds to be used for automatic multilevel global threshold determination
    - struc_elem_size: Size of disk - shaped structuring element to be used for morphological closing operation
    
    Outputs:
    - shadow_mask: Shadow mask for input image
    
    """
    
    if (convolve_window_size % 2 == 0):
        raise ValueError('Please make sure that convolve_window_size is an odd integer')
        
    buffer = int((convolve_window_size - 1) / 2)
    
    
    with rasterio.open(image_file) as f:
        metadata = f.profile
        img = rescale_intensity(np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0]), out_range = 'uint8')
        img = img[:, :, 0 : 3]
    
    
    lch_img = np.float32(lab2lch(rgb2lab(img)))
    
    
    l_norm = rescale_intensity(lch_img[:, :, 0], out_range = (0, 1))
    h_norm = rescale_intensity(lch_img[:, :, 2], out_range = (0, 1))
    sr_img = (h_norm + 1) / (l_norm + 1)
    log_sr_img = np.log(sr_img + 1)
    
    del l_norm, h_norm, sr_img
    gc.collect()

    

    avg_kernel = np.ones((convolve_window_size, convolve_window_size)) / (convolve_window_size ** 2)
    blurred_sr_img = cv2.filter2D(log_sr_img, ddepth = -1, kernel = avg_kernel)
      
    
    del log_sr_img
    gc.collect()
    
                
    flattened_sr_img = blurred_sr_img.flatten().reshape((-1, 1))
    labels = KMeans(n_clusters = num_thresholds + 1, max_iter = 10000).fit(flattened_sr_img).labels_
    flattened_sr_img = flattened_sr_img.flatten()
    df = pd.DataFrame({'sample_pixels': flattened_sr_img, 'cluster': labels})
    threshold_value = df.groupby(['cluster']).min().max()[0]
    df['Segmented'] = np.uint8(df['sample_pixels'] >= threshold_value)
    
    
    del blurred_sr_img, flattened_sr_img, labels, threshold_value
    gc.collect()
    
    
    shadow_mask_initial = np.array(df['Segmented']).reshape((img.shape[0], img.shape[1]))
    struc_elem = disk(struc_elem_size)
    shadow_mask = np.expand_dims(np.uint8(cv2.morphologyEx(shadow_mask_initial, cv2.MORPH_CLOSE, struc_elem)), axis = 0)
    
    
    del df, shadow_mask_initial, struc_elem
    gc.collect()
    

    return shadow_mask

def type_of_building(x):
    area, height = x[0],x[1]

    if height<6 or  pd.isna(height):
        if area<250:
            building_type='menage individuel'
            count=1
            return building_type,count
        else:
            building_type='menage collectif-cour commune'
            count=area//100
            return building_type,count
    else:
        building_type='menage collectif-etage'
        count=(area//100)*(height//3)
        return building_type,count



def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print("Done")

create_dir('results')




if __name__ == "__main__":
    arg = sys.argv[:]
    if len(arg) < 4:
        print('Not enough arguments!\n')
        print('python height_estimation.py geojson_url tif_url img_dir azimuth out_path')
        exit(0)

    building_bingervile_path = str(arg[1])
    rgb_bingerville = str(arg[2])
    bingerville_buildding_dir = str(arg[3])
    azimuth = str(arg[4])
    out_path = str(arg[5])

    building_bingerville = gpd.read_file(building_bingervile_path)

    condition = ''
    grid_num = '042'
    CLASSIFY = Path('results')
    CLASSIFY.mkdir(exist_ok=True)
    (CLASSIFY/f'{condition}_{grid_num}').mkdir(exist_ok=True)

    for i,row in tqdm(building_bingerville[(building_bingerville['geometry'].type=='Polygon')].iterrows()):
        
        poly = row['geometry'].buffer(0.00001) # padding around detection to crop
    #     print(poly.bounds)

        inProj = Proj(init='epsg:4326') 
        outProj = Proj(init=raster_rgb_bingerville.meta['crs']['init']) # convert to cog crs
        
        # convert from geocoords to display window
        minx, miny = transform(inProj,outProj,*poly.bounds[:2])
        maxx, maxy = transform(inProj,outProj,*poly.bounds[2:])
        ul = raster_rgb_bingerville.index(minx, miny)
        lr = raster_rgb_bingerville.index(maxx, maxy)
        disp_minx, disp_maxx, disp_miny, disp_maxy = lr[0], (max(ul[0],0)+1), max(ul[1],0), (lr[1]+1)

        if disp_maxx-disp_minx <= 150: disp_maxx += 25; disp_minx-=25; 
        if disp_maxy-disp_miny <= 150: disp_maxy += 25; disp_miny-=25;

        try:
          window = (max(disp_minx,0), disp_maxx), (max(disp_miny,0), disp_maxy)
          data = raster_rgb_bingerville.read(window=window)
          
          # pk = str(row).zfill(5)
          tile_bgr = cv2.cvtColor(np.rollaxis(data,0,3), cv2.COLOR_RGB2BGR)
          cv2.imwrite(f"{str(CLASSIFY)}/{condition}_{grid_num}/{grid_num}_{i}_{condition}.jpg", tile_bgr)
        except:
          pass


    building_bingerville['area'] = (building_bingerville.to_crs({'init': 'epsg:3395'} )).geometry.map(lambda p: p.area )#/ 10**6)  

    imagepaths = []
    # for dirpath, dirnames, filenames in os.walk(r"F:\intership_files\abidjan quicbird\BINGERVILLE"):
    for dirpath, dirnames, filenames in os.walk(bingerville_buildding_dir):
        for filename in filenames:
            if filename.endswith('.jpg'):
            # print(filename)
                img = cv2.imread(os.path.join(dirpath, filename))
                if np.array_equiv(img, np.zeros_like(img))==False:
                    imagepaths.append(os.path.join(dirpath, filename))
            
    N = len(imagepaths)

    pi=22/7
    degree = float(azimuth)#152.71)#input("Input degrees: "))
    radian = degree*(pi/180)



    df_building_list = []
    j = 0
    for file in imagepaths:
        gray = shadow_detection(file, convolve_window_size = 5, num_thresholds = 3, struc_elem_size = 5)# cv2.imread(file)
        cnts = cv2.findContours( reshape_as_image(gray).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#cv2.CHAIN_APPROX_NONE
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        area_height = []
        area_width = []
        area_shadow = []
        position_x = []
        position_y = []
        file_list = []
        for c in cnts:
            area = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)
            r_h = 2.5*h*np.tan(radian)
            r_w = 2.5*w*np.tan(radian)
            if 723.0 < area <40000:

                area_height.append(h)
                area_width.append(w)
                area_shadow.append(area)
                position_x.append(x)
                position_y.append(y)

        df_building = pd.DataFrame({'building_height':area_height,'building_width':area_width,'area':area_shadow,'x':position_x, 'y':position_y})
        df_building['real_height'] =  2.5*df_building["building_height"]*np.abs(np.tan(radian))
        df_building['real_width'] =  2.5*df_building["building_width"]*np.abs(np.tan(radian))
        filename = file.replace('\\', '/').split('/')[-1]
        if len(area_shadow)!=0:
            df_building['id'] = [int(filename.split('_')[1])]*len(area_shadow)
        else:
            df_building['id'] = [int(filename.split('_')[1])]


        df_building_list.append(df_building)
        j +=1


        prog = ((j)/len(imagepaths)) * 100
        print('\rCompleted: {:.2f}%'.format(prog),end=' ')

    dfs_building = pd.concat(df_building_list, axis=0)


    building_bingerville["id"]=building_bingerville.index

    height_building_bingerville = pd.merge(building_bingerville, dfs_building.drop(['area'], axis=1), on=['id'])

    height_building_bingerville["building_type"] = building_bingerville.loc[:,["area","building_height"]].apply(type_of_building, axis=1).apply(lambda x:x[0])
    height_building_bingerville["building_type"] = height_building_bingerville.loc[:,["area","building_height"]].apply(type_of_building, axis=1).apply(lambda x:x[0])#.unique()
    height_building_bingerville["household_count"] = height_building_bingerville.loc[:,["area","building_height"]].apply(type_of_building, axis=1).apply(lambda x:x[1])#.unique()

    mi = height_building_bingerville[height_building_bingerville["building_type"]=='menage individuel']["household_count"].sum()
    mcc = height_building_bingerville[height_building_bingerville["building_type"]=='menage collectif-cour commune']["household_count"].sum()
    mce = height_building_bingerville[height_building_bingerville["building_type"]=='menage collectif-etage']["household_count"].sum()


    df1 = height_building_bingerville[["building_type","household_count"]]


height_building_bingerville = height_building_bingerville.drop(['geometry_buffered','coords_pixel'],axis=1)
with open(out_path, 'w') as f:
  f.write(height_building_bingerville.to_json())

