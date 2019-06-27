import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import math


def load_image(path, img_list, img_height, img_width):
    ''' This function loads and downsamples images for training.
        Output: A Numpy 2D matrix that stores every training sample in rows. '''
    
    img_size = img_height * img_width
    mat_images = np.zeros((len(img_list), img_size))    
    
    for i, file in enumerate(img_list):
        img = Image.open(path + file)
        img = img.resize((img_height, img_width), Image.LANCZOS)
        mat_images[i] = np.asarray(img).flatten()

    return mat_images
    
    
def get_centers(dir_roi, resol=(2048, 2048)):
    ''' This function scans through the directory of the ROI files, reads the
        coordinates (in pixel) of the centers of all circular ROIs, then
        transforms them into relative coordinates (assumming image size 1x1) '''  

    img_id = []
    cx_pxl = []
    cy_pxl = []

    HEIGHT, WIDTH = resol
    
    # Get center coordinates of all sample images
    for dirpath, dirnames, files in os.walk(dir_roi):
        sample_id = os.path.split(dirpath)[1].split('.')[0]
        for file in files:
            img_id.append(sample_id + file[:4])
            cx_pxl.append(int(file[10:14]))
            cy_pxl.append(int(file[5:9]))

    df_roi = pd.DataFrame({'img_id': img_id,
                           'cx_pxl': cx_pxl,
                           'cy_pxl': cy_pxl},
                           columns = ['img_id', 'cx_pxl', 'cy_pxl'])
    df_roi['cx'] = df_roi.cx_pxl / float(WIDTH)
    df_roi['cy'] = df_roi.cy_pxl / float(HEIGHT)
            
    return df_roi


def coord_transfm(df_roi_org, shifts=(359, 359), cropped_resol=(1330, 1330)):
    shift_x, shift_y = shifts
    HEIGHT, WIDTH = cropped_resol
    df_roi = df_roi_org.copy()
    df_roi.cx_pxl = df_roi.cx_pxl - shift_x
    df_roi.cy_pxl = df_roi.cy_pxl - shift_y
    df_roi.cx = df_roi.cx_pxl / float(WIDTH)
    df_roi.cy = df_roi.cy_pxl / float(HEIGHT)
    
    return df_roi 


def mirror(df_imgs, flip_axis, img_height, img_width):
    
    pxl_cols = [col for col in df_imgs.columns if 'pxl' in col]
    num_imgs = df_imgs.shape[0]
    img_size = img_height * img_width
    
    mat_images = df_imgs[pxl_cols].values
    mat_images = mat_images.reshape((num_imgs, img_height, img_width))
    df_imgs_flip = df_imgs.copy()
    
    assert 'h' in flip_axis or 'v' in flip_axis, \
           "Flipping axis is not defined. Must be either 'horizontal' or 'vertical'."
    
    if 'h' in flip_axis:
        flipping = 1
        df_imgs_flip.img_id = df_imgs.img_id + 'hf'
        df_imgs_flip.cy = 1.0 - df_imgs.cy  
    elif 'v' in flip_axis:
        flipping = 2
        df_imgs_flip.img_id = df_imgs.img_id + 'vf'
        df_imgs_flip.cx = 1.0 - df_imgs.cx    
    
    # Flip images around the specified axis    
    mat_images_flip = np.flip(mat_images, flipping) \
                        .reshape((num_imgs, img_size))
    df_imgs_flip[pxl_cols] = mat_images_flip
    
    return df_imgs_flip
  
    
   
    
                


