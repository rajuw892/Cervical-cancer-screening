#!/usr/bin/env python3
#Quiero utf8: áéíóú

from PIL import ImageFilter, ImageStat, Image, ImageDraw
from multiprocessing import Pool, cpu_count
# conda install --channel https://conda.anaconda.org/menpo opencv3
from cv2 import imread as cv2_imread, resize as cv2_resize, INTER_AREA as cv2_INTER_AREA # http://tanbakuchi.com/posts/comparison-of-openv-interpolation-algorithms/

import time
def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f.__name__, ': ', '{:,.4f}'.format(end - start), ' segs.')
        return(result)
    return(f_timer)

from datetime import datetime
def jj_datetime(): return(datetime.now().strftime('%Y-%m-%d %H:%M:%S -')) # print(jj_datetime(), xxxx)
def jj_datetime_filename(): return(datetime.now().strftime('%Y%m%d_%H%M%S'))

def jj_input_filename_suffix(n_resize_to, b_con_muestreo): return('_{:}'.format(n_resize_to) + ('_prueba' if b_con_muestreo else ''))

def jj_safe_exec(ret_if_exception, function, *args):
    try:
        return(function(*args))
    except:
        return(ret_if_exception)

def im_multi(path):
    from PIL import ImageFilter, ImageStat, Image, ImageDraw
    try:
        im_stats_im_ = Image.open(path)
        return [path, {'size': im_stats_im_.size}]
    except:
        print(path)
        return [path, {'size': (0,0)}]

@timefunc
def im_stats(im_stats_df):
    from multiprocessing import Pool, cpu_count
    im_stats_d = {}
    p = Pool(cpu_count() - 1)
    #ret = [p.apply_async(im_multi, x) for x in im_stats_df['path']] # Y luego hay que usar ret[n].get() para sacar cada resultado!
    ret = p.map(im_multi, im_stats_df['path'])
    for i in range(len(ret)):
        im_stats_d[ret[i][0]] = ret[i][1] # im_stats_d[ret[i].get()[0]] = ret[i].get()[1]
    im_stats_df['size'] = im_stats_df['path'].map(lambda x: ' '.join(str(s) for s in im_stats_d[x]['size']))
    return im_stats_df

def get_im_cv2_32(path):
    img = cv2_imread(path)
    resized = cv2_resize(img, (32, 32), cv2_INTER_AREA) #use cv2_resize(img, (64, 64), cv2_INTER_AREA)
    return [path, resized]

def get_im_cv2_64(path):
    img = cv2_imread(path)
    resized = cv2_resize(img, (64, 64), cv2_INTER_AREA)
    return [path, resized]

def get_im_cv2_256(path):
    img = cv2_imread(path)
    resized = cv2_resize(img, (256, 256), cv2_INTER_AREA)
    return [path, resized]

def get_im_cv2_512(path):
    img = cv2_imread(path)
    resized = cv2_resize(img, (512, 512), cv2_INTER_AREA)
    return [path, resized]

def get_im_cv2_1024(path):
    img = cv2_imread(path)
    resized = cv2_resize(img, (1024, 1024), cv2_INTER_AREA)
    return [path, resized]

@timefunc
def normalize_image_features(paths, resize_to = 32):
    import numpy as np
    imf_d = {}
    p = Pool(cpu_count())
    if resize_to == 256:
        ret = p.map(get_im_cv2_256, paths)
    elif resize_to == 64:
        ret = p.map(get_im_cv2_64, paths)
    elif resize_to == 512:
        ret = p.map(get_im_cv2_512, paths)
    elif resize_to == 1024:
        ret = p.map(get_im_cv2_1024, paths)
    else:
        ret = p.map(get_im_cv2_32, paths)
    
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    fdata = np.array(fdata, dtype=np.uint8)
    fdata = fdata.transpose((0, 3, 1, 2))
    fdata = fdata.astype('float32') # fdata.astype('float64')
    fdata = fdata / 255
    return fdata

