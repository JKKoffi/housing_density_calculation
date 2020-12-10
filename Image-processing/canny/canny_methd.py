"""
Title: Rooftops detection  Image using canny method
Author: Jonathan KOFFI
Date created: 2020
Last Modified: 10/23/2020
"""
import numpy as np
import cv2
import os
from time import time
import sys

#from shapely.geometry import MultiPolygon, Polygon

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print("Done")

res_dir = "tiles_res"
res_on_im_dir = "tiles_res_img"
create_dir(res_dir)
create_dir(res_on_im_dir)



def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # edged = cv2.Canny(image, lower, upper,apertureSize=3)
    # return the edged image
    return edged

def sharpen(img):
    # SHARPEN
    kernel_sharp = np.array(([-2, -2, -2], [-2, 17, -2], [-2, -2, -2]), dtype='int')
    # ernel_sharp = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), dtype='int')
    return cv2.filter2D(img, -1, kernel_sharp)


# def equalize(img):
#     ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
#     channels = cv2.split(ycrcb)
#     cv2.equalizeHist(channels[0], channels[0])
#     cv2.merge(channels, ycrcb)
#     cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
#     return img

# subgray = gray[1250:1500,500:1000]    
# sub_bgrimg = bgrimg[1250:1500,500:1000,:]
# sub_markers1 = markers2.copy()
# markers2 = np.zeros_like(bgrimg)
# markers2 = markers2[sub_markers1!=0]

def mask_roofs(bgrimg):
  # GET IMAGE AND RESIZE
  gray = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2GRAY)
  # plt.imshow(gray,cmap='gray'),plt.title('gray image'),plt.show()
  # SHARPEN
  sgray = sharpen(gray)
  # plt.imshow(sgray,cmap='gray'),plt.title('shapened gray image'),plt.show()
  sbgrimg = sharpen(bgrimg)
  # plt.imshow(sbgrimg,cmap='gray'),plt.title('shapened gray image'),plt.show()

  # THRESHOLDING
  ret, mask = cv2.threshold(sgray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  # ret, mask = cv2.threshold(sgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  # plt.imshow(mask,cmap='gray'),plt.title('threshed image'),plt.show()

  # EDGES
  edges = auto_canny(mask)
  # plt.imshow(edges,cmap='gray'),plt.title('auto canny image'),plt.show()
  invedges = cv2.bitwise_not(edges)
  # plt.imshow(invedges,cmap='gray'),plt.title('auto canny image'),plt.show()

  # REFINE MASK
  mieg = cv2.bitwise_and(mask, invedges)
  kernel = np.ones((3,3), np.uint8)
  opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
  refined = cv2.bitwise_and(mieg, opening)
  # refined = cv2.bitwise_not(refined)
  # plt.imshow(refined,cmap='gray'),plt.title('refined image'),plt.show()

  # CONVERT MASK TO MATCH WITH ORIGNAL IMAGE DIMENSIONS
  refined3d = sbgrimg.copy()
  # refined3d = np.zeros_like(sbgrimg)
  vidx, hidx = refined.nonzero()
  for ii in range(len(vidx)):
      refined3d[vidx[ii]][hidx[ii]][0] = 0
      refined3d[vidx[ii]][hidx[ii]][1] = 255
      refined3d[vidx[ii]][hidx[ii]][2] = 255
  
  markers2 = np.zeros_like(sbgrimg)
    # DRAW CONTOURS
  buildings = 0
    # img2 = np.zeros_like(markers2)
  contours, hierarchy = cv2.findContours(refined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  w_max = 0
  h_max = 0
  for contour in contours:
    area = cv2.contourArea(contour)
    x,y,w,h = cv2.boundingRect(contour)
    # print(x,y,w,h)
    # if area > 2000:
    if (area > 200 ) and (w<109.) and (h<100. ):
        # if w<109. and h>20:
            # cv2.drawContours(bgrimg, contour, -1, (0, 255, 0), 2)
            cv2.fillPoly(markers2, pts =[contour], color=(0,255,255))
            # bgrimg = cv2.polylines(bgrimg,[contour],True,(0,255,255))
            buildings += 1 
            # print(x,y,w,h)
            if w>w_max:
                w_max=w
            if h>h_max:
                h_max=h
    print(h_max,w_max)
            # cv2.drawContours(img2,contour,-1,(255,255,255),5)
    # plt.imshow(markers2),plt.title('canny edge detection'),plt.show()
    # return refined3d
    return buildings,markers2,refined3d

# src:https://stackoverflow.com/questions/33046734/difficult-time-trying-to-do-shape-recognition-for-3d-objects
def denosing(img_1ch):
    # Implementing morphological erosion & dilation
    kernel = np.ones((3,3),np.uint8)  # (6,6) to get more contours (9,9) to reduce noise
    opening = cv2.erode(mask, kernel, iterations = 1)
    closing = cv2.dilate(opening, kernel, iterations=1)
    return closing

def canny_met(img_path):

    start = time()

    bgrimg = cv2.imread(img_path)
    # plt.imshow(bgrimg),plt.show()           
    # bgrimg = bgrimg[:400,1300:,:]

    buildings,mask,refined3d = mask_roofs(bgrimg)
    sbgrimg = sharpen(bgrimg)
    
    stacked3d = np.hstack((sbgrimg, refined3d))

    elapsed_time = time() - start

    filename = os.path.split(img_path)[1]
    mask_path = os.path.join(res_dir,filename)
    scoring_path = os.path.join(res_dir,filename.replace('.tif','.csv'))
    overlay_img_path = os.path.join(res_on_im_dir,filename)

    # plt.imshow(sbgrimg, cmap='gray'),plt.show()

    # plt.imshow(img2, cmap='gray'),plt.show()
    df = pd.DataFrame({'elapsed_time':[elapsed_time],'numbers':[buildings]})
    df.to_csv(scoring_path, index=False)
    cv2.imwrite(mask_path, mask)
    #write detected building on theoriginal image
    cv2.imwrite(overlay_img_path, stacked3d)


if __name__ == "__main__":
    arg = sys.argv[:]
    if len(arg) < 2:
        print('Not enough arguments!\n')
        print('python canny_methd.py path_to_images path_to_result path_to_result1')
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
            canny_met(imgpath)
        except:
            pass
        prog = ((i+1)/len(imagepaths)) * 100
        print('\rCompleted: {:.2f}%'.format(prog),end=' ')

