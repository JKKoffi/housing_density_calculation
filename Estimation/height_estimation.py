# building height estimation withshadow length

import cv2
import gc
import numpy as np
import pandas as pd
import rasterio
from skimage.color import lab2lch, rgb2lab
from skimage.exposure import rescale_intensity
from skimage.morphology import disk
from sklearn.cluster import KMeans
from rasterio.transform import *
from shapely.geometry import *
import numpy as np

import rasterio as rio
from rasterio.plot import *

pi=22/7

def shadow_detection(image_file, shadow_mask_file, convolve_window_size = 5, num_thresholds = 3, struc_elem_size = 5):
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
    

    metadata['count'] = 1
    with rasterio.open(shadow_mask_file, 'w', **metadata) as dst:
        dst.write(shadow_mask)
        
    return shadow_mask

def shadow_correction(image_file, shadow_mask_file, corrected_image_file, exponent = 1):
    """
    This function is used to adjust brightness for shadow - covered areas in an image, as proposed in the paper 
    'Near Real - Time Shadow Detection and Removal in Aerial Motion Imagery Application' by Silva G.F., Carneiro G.B., 
    Doth R., Amaral L.A., de Azevedo D.F.G. (2017)
    
    Inputs:
    - image_file: Path of 3 - channel (red, green, blue) image to be processed for shadow removal
    - shadow_mask_file: Path of shadow mask for corresponding input image
    - corrected_image_file: Path of corrected image to be saved
    - exponent: Exponent to be used for the calculcation of statistics for unshaded and shaded areas
    
    Outputs:
    - corrected_img: Corrected input image
    
    """
    
    with rasterio.open(image_file) as f:
        metadata = f.profile
        img = rescale_intensity(np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0]), out_range = 'uint8')
        
    with rasterio.open(shadow_mask_file) as s:
        shadow_mask = s.read(1)
    
    corrected_img = np.zeros((img.shape), dtype = np.uint8)
    non_shadow_mask = np.uint8(shadow_mask == 0)
    
    
    for i in range(img.shape[2]):
        shadow_area_mask = shadow_mask * img[:, :, i]
        non_shadow_area_mask = non_shadow_mask * img[:, :, i]
        shadow_stats = np.float32(np.mean(((shadow_area_mask ** exponent) / np.sum(shadow_mask))) ** (1 / exponent))
        non_shadow_stats = np.float32(np.mean(((non_shadow_area_mask ** exponent) / np.sum(non_shadow_mask))) ** (1 / exponent))
        mul_ratio = ((non_shadow_stats - shadow_stats) / shadow_stats) + 1
        corrected_img[:, :, i] = np.uint8(non_shadow_area_mask + np.clip(shadow_area_mask * mul_ratio, 0, 255))
    

    with rasterio.open(corrected_image_file, 'w', **metadata) as dst:
        dst.write(np.transpose(corrected_img, [2, 0, 1]))
        
    return corrected_img


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print("Done")

#shadow removal
shadow_mask_dir = "results/mask_shadow"
img_sgm = "results/img_segm"


create_dir(shadow_mask_dir)
create_dir(img_sgm)



def height_estimation(image_file, shadow_mask_file,azimuth, df_path, sgm_path):

    degree = float(azimuth)#152.71)#input("Input degrees: "))
    radian = degree*(pi/180)

    mask = shadow_detection(image_file, shadow_mask_file, convolve_window_size = 5, num_thresholds = 3, struc_elem_size = 5)
    mask_img =  reshape_as_image(mask)
    src = rio.open(image_file)

    img = cv2.imread(image_file)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gray = mask_img# reshape_as_image(res)   
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#cv2.CHAIN_APPROX_NONE
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    from copy import deepcopy
    image = deepcopy(img)
    building_height = []
    building_width = []
    area_shadow = []
    position_x = []
    position_y = []
    #segmentation of the mask
    for c in cnts:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        r_h = 2.5*h*np.tan(radian)
        r_w = 2.5*w*np.tan(radian)
        if 723.0 < area <40000:

            building_height.append(h)
            building_width.append(w)
            area_shadow.append(area)
            # roi=X[y:y+h,x:x+w]
            # cv2.imwrite(f'results1/{buildings}' + '.jpg', roi)
            # cv2.drawContours(image, [c], -1, (36,255,12), 4)
            cv2.drawContours(image, [c], -1, (255,255,0), -1) #wiill draw full line on objects | mask
            # cv2.putText(image,f"({w},{h})", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)

            position_x.append(x)
            position_y.append(y)


    df_building = pd.DataFrame({'building_height':building_height,'building_width':building_width,'area':area_shadow,'x':position_x, 'y':position_y})
    df_building.head()


    # print(radian)
    df_building['real_height'] =  2.5*df_building["building_height"]*np.abs(np.tan(radian))
    df_building['real_width'] =  2.5*df_building["building_width"]*np.abs(np.tan(radian))

    df_building['real_x'] = src.xy(df_building['x'],df_building['x'])[0]
    df_building['real_y'] = src.xy(df_building['x'],df_building['x'])[1]

    df_building.to_csv(df_path, index=False, header=True)

    cv2.imwrite(sgm_path, image)




if __name__ == "__main__":
    arg = sys.argv[:]
    if len(arg) < 4:
        print('Not enough arguments!\n')
        print('python height_estimation.py img_dir ext azimuth')
        exit(0)

    img_dir = str(arg[1])
    ext = str(arg[2])
    azimuth = float(arg[3])

    imagepaths = []
    for dirpath, dirnames, filenames in os.walk(img_dir):
        for filename in filenames:
            if filename.endswith(ext):
            # print(filename)
                imagepaths.append(os.path.join(dirpath, filename))
            
    N = len(imagepaths)
    j = 0
    for file in imagepaths:
        fname = file.replace('\\','/').split('/')[-1]
        shadow_mask_file = os.path.join(shadow_mask_dir,'mas_'+fname)
        df_path = os.path.join(shadow_mask_dir,'df_'+fname.replace(ext, '.csv'))
        imagefile =  os.path.join(img_sgm,fname)

        height_estimation(file, shadow_mask_file,azimuth, df_path, imagefile)
        j +=1
        prog = (j/N) * 100
        print('\rCompleted: {:.2f}%'.format(prog),end=' ')




