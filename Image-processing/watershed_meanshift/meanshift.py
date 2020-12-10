from __future__ import print_function
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from time import time
import numpy as np
import cv2
import os
from time import time

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print("Done")

res_dir = "tiles_res"
res_on_im_dir = "tiles_res_img"
create_dir(res_dir)
create_dir(res_on_im_dir)



def equalize(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img
def pyMeanshift_methd(img_path):
  
  start = time()
  
  image = cv2.imread(img_path)
  image2 = image.copy()
  im = equalize(image)
  shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
  
  # convert the mean shift image to grayscale, then apply
  gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  
  D = ndimage.distance_transform_edt(thresh)
  localMax = peak_local_max(D, indices=False, min_distance=20,labels=thresh)
  
  # perform a connected component analysis on the local peaks,
  # using 8-connectivity, then appy the Watershed algorithm
  markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
  labels = watershed(-D, markers, mask=thresh)
  print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
  
  # loop over the unique labels returned by the Watershed
  markers2 = np.zeros_like(image, dtype='uint8')
  w_max = 0
  h_max = 0
  buildings = 0
  # algorithm
  for label in np.unique(labels):
    # if the label is zero, we are examining the 'background'
    # so simply ignore it
    
    if label == 0:
      continue
    # otherwise, allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
    
    # detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # c = max(cnts, key=cv2.contourArea)
    
    # draw a circle enclosing the object
    # ((x, y), r) = cv2.minEnclosingCircle(c)
    # cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
    # cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
    # cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # n=0
    # loop over the contours
    for (i, c) in enumerate(cnts):
      area = cv2.contourArea(c)
      x,y,w,h = cv2.boundingRect(c)
      if (w <109  ) and  (h<100. ) and (area > 200) and area <1000 :#and (w<109.) and (h<100. ):
          
          buildings += 1 
          
          # # print(x,y,w,h)
          # if w>w_max:
          #     w_max=w
          # if h>h_max:
          #     h_max=h
          # print(w_max,h_max)
          # draw the contour
          ((x, y), _) = cv2.minEnclosingCircle(c)
          # cv2.putText(image2, "#{}".format(i + 1), (int(x) - 10, int(y)),
          #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
          cv2.drawContours(image2, [c], -1, (0, 255, 0), 2)
          # cv2.fillPoly(markers2, [c], -1, (0, 255, 0), 2)
          cv2.fillPoly(markers2, pts =[c], color=(255,255,255))
  
  stacked3d = np.hstack((image, image2))
  elapsed_time = time() - start
  filename = os.path.split(img_path)[1]
  mask_path = os.path.join(res_dir,filename)
  scoring_path = os.path.join(res_dir,filename.replace('.tif','.csv'))
  overlay_img_path = os.path.join(res_on_im_dir,filename)
  
  df = pd.DataFrame({'elapsed_time':[elapsed_time],'numbers':[buildings]})
  df.to_csv(scoring_path, index=False)
  cv2.imwrite(mask_path, markers2)
  #write detected building on theoriginal image
  cv2.imwrite(overlay_img_path, stacked3d)



if __name__ == "__main__":
    arg = sys.argv[:]
    if len(arg) < 2:
        print('Not enough arguments!\n')
        print('python meanshift.py path_to_images  ')
        exit(0)

    imagepaths = []
    # for dirpath, dirnames, filenames in os.walk(r"F:\intership_files\abidjan quicbird\BINGERVILLE"):
    for dirpath, dirnames, filenames in os.walk(str(arg[1])):
        for filename in filenames:
            if filename.endswith('.tif'):
            # print(filename)
                imagepaths.append(os.path.join(dirpath, filename))
            

    # ITERATE OVER ALL IMAGES
    for i, imgpath in enumerate(imagepaths):
        try:
            pyMeanshift_methd(imgpath)
        except:
            pass
        prog = ((i+1)/len(imagepaths)) * 100
        print('\rCompleted: {:.2f}%'.format(prog),end=' ')

