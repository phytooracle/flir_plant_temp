#!/usr/bin/env python3
"""
Author : Emmanuel Gonzalez
Date   : 2020-12-04
Purpose: Individual temperature extraction
"""

import argparse
import os
import sys
from detecto import core, utils, visualize
import random
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tifffile as tifi
import pandas as pd
import copy
from scipy import stats
from scipy.signal import argrelextrema
import skimage.color
from skimage.filters import threshold_minimum, threshold_triangle, threshold_yen, try_all_threshold
from skimage import data, exposure, img_as_float
import multiprocessing
import seaborn as sns

# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Individual plant temperature extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dir',
                        nargs='+',
                        metavar='dir',
                        help='Directory containing geoTIFFs')

    parser.add_argument('-m',
                        '--model',
                        help='Trained model (.pth)',
                        metavar='model',
                        type=str,
                        required=True)

    parser.add_argument('-od',
                        '--outdir',
                        help='Output directory',
                        metavar='outdir',
                        type=str,
                        default='individual_thermal_out')

    parser.add_argument('-c',
                        '--cpu',
                        help='Number of CPUs for multiprocessing',
                        metavar='cpu',
                        type=int,
                        required=True)

    parser.add_argument('-of',
                        '--outfile',
                        help='Output filename',
                        metavar='outfile',
                        default='individual_temps')

    return parser.parse_args()


# --------------------------------------------------
def open_image(img_path):

    tif_img = tifi.imread(img_path)
    a_img = cv2.cvtColor(tif_img, cv2.COLOR_GRAY2BGR)
    a_img = cv2.normalize(a_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return a_img, tif_img


# --------------------------------------------------
def peak_temp(clipped_img):

    sns_distplot = sns.distplot(clipped_img.ravel(), hist=True, kde=True, bins=int(a_img.max()/1), color='darkblue').get_lines()[0].get_data()
    x_y = (np.stack((sns_distplot[0], sns_distplot[1]), axis=1))
    minima = argrelextrema(sns_distplot[1], np.less_equal)
    a_img_copy = clipped_img.copy()

    minima_array = sns_distplot[0][minima]

    if len(minima_array) > 1:
        minima_val = float(minima_array[1])
    cut_off = float(minima_val)
    # print(f'Original cut off: {cut_off}')

    if cut_off < 300:
        try:
            minima_val = float(minima_array[2])
            cut_off = float(minima_val)
            # print(cut_off)
        except:
            cut_off = cut_off
            # print(f'New cut off: {cut_off}')

    a_img_copy[a_img_copy > cut_off] = np.nan

    plant_temp = np.nanmean(a_img_copy)

    return plant_temp


def min_threshold(image, sigma = float(0.5)):

    blur = skimage.color.rgb2gray(image)
    blur = skimage.filters.gaussian(image, sigma=sigma)
    t = skimage.filters.threshold_minimum(blur)

    mask = blur < t
    sel = np.zeros_like(image)
    sel[mask] = image[mask]
    sel[sel == 0] = np.nan

    mask2 = blur > t
    sel2 = np.zeros_like(image)
    sel2[mask2] = image[mask2]
    sel2[sel2 == 0] = np.nan

    soil_temp = np.nanmean(sel2)
    plant_temp = np.nanmean(sel)
    img_temp = np.nanmean(image)

    return plant_temp


# --------------------------------------------------
def process_image(img):

    try:
        temp_dict = {}
        cnt = 0
        args = get_args()

        model = core.Model.load(args.model, ['lettuce'])
        a_img, tif_img = open_image(img)
        predictions = model.predict(a_img)
        labels, boxes, scores = predictions
        copy = tif_img.copy()


        for i, box in enumerate(boxes):
            if scores[i] >= 0.2:
                cnt += 1
                min_x, min_y, max_x, max_y = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                area = (max_x - min_x) * (max_y - min_y)

                start_point = (min_x, max_y)
                end_point = (max_x, min_y)

                center_x = abs(min_x - max_x)//2
                center_y = abs(min_y - max_y)//2

                new_img = tif_img[min_y:max_y, min_x:max_x]
                new_img = np.array(new_img)
                copy = new_img.copy()

                e_p = (int(center_x-10), int(center_y+10))
                s_p = (int(center_x+10), int(center_y-10))

                roi = copy[int(center_y-10):int(center_y+10), int(center_x-10):int(center_x+10)]
                temp_roi = np.nanmean(roi)
                img_temp = np.nanmean(new_img)
                peak = peak_temp(new_img)
                min_thresh = min_threshold(new_img)

                f_name = img.split('/')[-1]
                temp_dict[cnt] = {
                                'image': f_name,
                                'roi_temp': temp_roi,
                                'image_temp': img_temp,
                                'peaks_temp': peak,
                                'min_thresh_temp': min_thresh
                                }

        df = pd.DataFrame.from_dict(temp_dict, orient='index', columns=['image',
                                                                        'roi_temp',
                                                                        'image_temp',
                                                                        'peaks_temp',
                                                                        'min_thresh_temp']).set_index('image')
    except:
        df = pd.DataFrame()

    return df


# --------------------------------------------------
def main():
    """Detect plants and collect temperatures here"""

    args = get_args()
    major_df = pd.DataFrame()

    with multiprocessing.Pool(args.cpu) as p:
        df = p.map(process_image, args.dir)
        major_df = major_df.append(df)

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    major_df.to_csv(os.path.join(args.outdir, f'{args.outfile}.csv'))

    print(f'Done, see outputs in ./{args.outdir}.')


# --------------------------------------------------
if __name__ == '__main__':
    main()
