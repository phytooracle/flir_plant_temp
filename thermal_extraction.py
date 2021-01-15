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
from osgeo import gdal
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
import utm
import json

import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Individual plant temperature extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dir',
                        metavar='dir',
                        help='Directory containing geoTIFFs')

    parser.add_argument('-m',
                        '--model',
                        help='Trained model (.pth)',
                        metavar='model',
                        type=str,
                        required=True)

    parser.add_argument('-g',
                        '--geojson',
                        help='GeoJSON containing plot boundaries',
                        metavar='str',
                        type=str,
                        default=None,
                        required=True)

    parser.add_argument('-d',
                        '--date',
                        help='Scan date',
                        metavar='date',
                        type=str,
                        default=None,
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
def get_paths(directory):
    ortho_list = []

    for root, dirs, files in os.walk(directory):
        for name in files:
            if '_ortho.tif' in name:
                ortho_list.append(os.path.join(root, name))

    if not ortho_list:

        raise Exception(f'ERROR: No compatible images found in {directory}.')


    print(f'Images to process: {len(ortho_list)}')

    return ortho_list


# --------------------------------------------------
def open_image(img_path):

    tif_img = tifi.imread(img_path)
    a_img = cv2.cvtColor(tif_img, cv2.COLOR_GRAY2BGR)
    a_img = cv2.normalize(a_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return a_img, tif_img


# --------------------------------------------------
def get_min_max(box):
    min_x, min_y, max_x, max_y = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    return min_x, min_y, max_x, max_y


# --------------------------------------------------
def get_trt_zones():
    trt_zone_1 = []
    trt_zone_2 = []
    trt_zone_3 = []

    for i in range(3, 19):
        for i2 in range(2, 48):
            plot = f'MAC_Field_Scanner_Season_10_Range_{i}_Column_{i2}'
            trt_zone_1.append(str(plot))

    for i in range(20, 36):
        for i2 in range(2, 48):
            plot = f'MAC_Field_Scanner_Season_10_Range_{i}_Column_{i2}'
            trt_zone_2.append(str(plot))

    for i in range(37, 53):
        for i2 in range(2, 48):
            plot = f'MAC_Field_Scanner_Season_10_Range_{i}_Column_{i2}'
            trt_zone_3.append(str(plot))

    return trt_zone_1, trt_zone_2, trt_zone_3


# --------------------------------------------------
def find_trt_zone(plot_name):
    trt_zone_1, trt_zone_2, trt_zone_3 = get_trt_zones()

    if plot_name in trt_zone_1:
        trt = 'treatment 1'

    elif plot_name in trt_zone_2:
        trt = 'treatment 2'

    elif plot_name in trt_zone_3:
        trt = 'treatment 3'

    else:
        trt = 'border'

    return trt


# --------------------------------------------------
def get_genotype(plot, geojson):

    with open(geojson) as f:
        data = json.load(f)

    for feat in data['features']:
        if feat.get('properties')['ID']==plot:
            genotype = feat.get('properties').get('genotype')

    return genotype


# --------------------------------------------------
def pixel2geocoord(one_img, x_pix, y_pix):
    ds = gdal.Open(one_img)
    c, a, b, f, d, e = ds.GetGeoTransform()
    lon = a * int(x_pix) + b * int(y_pix) + a * 0.5 + b * 0.5 + c
    lat = d * int(x_pix) + e * int(y_pix) + d * 0.5 + e * 0.5 + f

    return (lat, lon)


# --------------------------------------------------
def peak_temp(clipped_img):

    sns_distplot = sns.distplot(clipped_img.ravel(), hist=True, kde=True, bins=int(clipped_img.max()/1), color='darkblue').get_lines()[0].get_data()
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


# --------------------------------------------------
def min_threshold(image, sigma = float(1)):

    #blur = skimage.color.rgb2gray(image)
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
def roi_temp(img, max_x, min_x, max_y, min_y):

    center_x = abs(min_x - max_x)//2
    center_y = abs(min_y - max_y)//2
    roi = img[int(center_y-10):int(center_y+10), int(center_x-10):int(center_x+10)]

    temp_roi = np.nanmean(roi)

    return temp_roi


# --------------------------------------------------
def get_stats(img):
    img = img[~np.isnan(img)]

    mean = np.mean(img) #- 273.15
    median = np.percentile(img, 50)

    q1 = np.percentile(img, 25)
    q3 = np.percentile(img, 75)

    var = np.var(img)
    sd = np.std(img)

    return mean, median, q1, q3, var, sd


# --------------------------------------------------
def kmeans_temp(img):
    pixel_vals = img.reshape((-1,1))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    k = 3
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((img.shape))
    low_thresh = np.unique(segmented_image)[:1][0]
    upp_thresh = np.unique(segmented_image)[1:2][0]
    img[segmented_image > upp_thresh] = np.nan

    mean, median, q1, q3, var, sd = get_stats(img)

    return mean, median, q1, q3, var, sd


# --------------------------------------------------
def process_image(img):

    temp_dict = {}
    cnt = 0
    args = get_args()

    model = core.Model.load(args.model, ['lettuce'])
    plot = img.split('/')[-1].replace('_ortho.tif', '')
    trt_zone = find_trt_zone(plot)
    plot_name = plot.replace('_', ' ')
    print(f'Image: {plot_name}')
    genotype = get_genotype(plot_name, args.geojson)

    a_img, tif_img = open_image(img)
    df = pd.DataFrame()

    try:
        predictions = model.predict(a_img)
        labels, boxes, scores = predictions

        for i, box in enumerate(boxes):
            if scores[i] >= 0.2:
                cnt += 1
                min_x, min_y, max_x, max_y = get_min_max(box)
                center_x, center_y = ((max_x+min_x)/2, (max_y+min_y)/2)

                nw_lat, nw_lon = pixel2geocoord(img, min_x, max_y)
                se_lat, se_lon = pixel2geocoord(img, max_x, min_y)

                nw_e, nw_n, _, _ = utm.from_latlon(nw_lat, nw_lon, 12, 'N')
                se_e, se_n, _, _ = utm.from_latlon(se_lat, se_lon, 12, 'N')

                area_sq = (se_e - nw_e) * (se_n - nw_n)
                lat, lon = pixel2geocoord(img, center_x, center_y)

                new_img = tif_img[min_y:max_y, min_x:max_x]
                new_img = np.array(new_img)
                #copy = new_img.copy()

                temp_roi = roi_temp(new_img, max_x, min_x, max_y, min_y)
                mean, median, q1, q3, var, sd = kmeans_temp(new_img)
                # img_temp = np.nanmean(new_img)
                # peak = peak_temp(new_img)
                # min_thresh = min_threshold(new_img)
                # print(temp_roi)

                #mean, median, q1, q3 = get_stats(copy)

                f_name = img.split('/')[-1]
                temp_dict[cnt] = {'date': args.date,
                                    'treatment': trt_zone,
                                    'plot': plot,
                                    'genotype': genotype,
                                    'lon': lon,
                                    'lat': lat,
                                    'min_x': min_x,
                                    'max_x': max_x,
                                    'min_y': min_y,
                                    'max_y': max_y,
                                    'nw_lat': nw_lat,
                                    'nw_lon': nw_lon,
                                    'se_lat': se_lat,
                                    'se_lon': se_lon,
                                    'bounding_area_m2': area_sq,
                                    'roi_temp': temp_roi,
                                    'quartile_1': q1,
                                    'mean': mean,
                                    'median': median,
                                    'quartile_3': q3,
                                    'variance': var,
                                    'std_dev': sd}

        df = pd.DataFrame.from_dict(temp_dict, orient='index', columns=['date', 'treatment', 'plot', 'genotype',
                                                                        'lon', 'lat', 'min_x', 'max_x', 'min_y',
                                                                        'max_y', 'nw_lat', 'nw_lon', 'se_lat',
                                                                        'se_lon', 'bounding_area_m2', 'roi_temp',
                                                                        'quartile_1', 'mean', 'median', 'quartile_3', 'variance', 'std_dev']).set_index('date')

    except:
        pass

    return df


# --------------------------------------------------
def main():
    """Detect plants and collect temperatures here"""

    args = get_args()
    major_df = pd.DataFrame()

    img_list = get_paths(args.dir)

    with multiprocessing.Pool(args.cpu) as p:
        df = p.map(process_image, img_list)
        major_df = major_df.append(df)

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    major_df.to_csv(os.path.join(args.outdir, f'{args.outfile}.csv'))

    print(f'Done, see outputs in ./{args.outdir}.')


# --------------------------------------------------
if __name__ == '__main__':
    main()
