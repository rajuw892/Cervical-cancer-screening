#!/usr/bin/env python3
#Quiero utf8: áéíóú
# NOTA: UNICODE - UTF8 - También! Si no hay acentos, conversiones -> UNICODE/UTF8 a UTF8 (ed. Unicode)
# NOTA: Para borrar todas la variables: %reset
# NOTA: Arrancar desde /home/jjtoharia/ con:
# source kaggle/bin/activate
# [Python]  - python -i kaggle/IntelCervicalCancerScreening/jjtz_intel_prep_datos.py   [b_con_muestreo [n_resize_to]]
# [pyspark] - spark-submit --driver-memory 4G kaggle/IntelCervicalCancerScreening/prep_datos_RDD.py   [b_con_muestreo [n_resize_to]]
b_Spark = False

#from PIL import ImageFilter, ImageStat, Image, ImageDraw
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import glob
#import cv2
#from cv2 import imread as cv2_imread, resize as cv2_resize, INTER_LINEAR as cv2_INTER_LINEAR

from jjtz_intel_funciones import *

# import os # os.listdir(path); os.remove(pathname); os.rename(old,new); isfile(pathname)
from os.path import isfile as os_path_isfile
from os.path import isdir as os_path_isdir
from os import chdir as os_chdir
# from os import remove as os_remove
# from os import rename as os_rename

if __name__ ==  '__main__': # Protección necesaria en Windows) para los multi-process...
  
  from sys import argv
  
  print(argv)
  b_con_muestreo = (int(argv[1]) == 1) if(len(argv) > 1)  else True  # Hacer muestreo (reducción) de las imágenes
  n_resize_to =     int(argv[2])       if(len(argv) > 2)  else 32    # Tamaño (cuadrado) de las imágenes redimensaionadas [256, 64 ó 32]
  
  print('Params: (b_con_muestreo, n_resize_to)=', (b_con_muestreo, n_resize_to))
  if(len(argv) < 3):
    print('\nERROR: Faltan params: [b_con_muestreo, n_resize_to]\n')
    quit()
  
  s_input_path = 'C:/Servidor/JJTZ_Sync/Kaggle_Intel-MobileODT/'
  if not os_path_isdir(s_input_path):
    s_input_path = s_input_path.replace('C:/Servidor/JJTZ_Sync', 'C:/Servidor/JJTZ-Sync')
  if not os_path_isdir(s_input_path):
    print('\nERROR: s_input_path [' + s_input_path + '] NO encontrado!\n')
    quit()
  
  # s_output_path = 'C:/Personal/Dropbox/Musica/Tmp_Kaggle/IntelCervicalCancerScreening/Out/'
  # if not os_path_isdir(s_output_path):
  #   s_output_path = s_output_path.replace('C:/Personal/Dropbox', 'C:/Users/Adriana/Dropbox')
  # if not os_path_isdir(s_output_path):
  #   print('\nERROR: s_output_path [' + s_output_path + '] NO encontrado!\n')
  #   quit()
  
  print(jj_datetime(), 'input_path = ' + s_input_path)
  # print(jj_datetime(), 'output_path = ' + s_output_path)
  
  # No hace falta... os_chdir(s_input_path) # Necesario para leer los ficheros...
  train = glob.glob(s_input_path + 'train/**/*.jpg') # + glob.glob(s_input_path + 'additional/**/*.jpg')
  train = [x.replace('\\', '/') for x in train] # Quitamos s_input_path y cambiamos \ por / [Windows!]
  train = pd.DataFrame([[p.split('/')[-2],p.split('/')[-1],p] for p in train], columns = ['type','image','path'])
  print(jj_datetime(), train.shape)
  if b_con_muestreo:
    train = train[::10] #limit for Kaggle Demo (seleccionamos uno de cada X registros)
    print(jj_datetime(), 'After sampling:', train.shape)
  
  #quit() - Corregido con if __name__ ==  '__main__'
  
  print(jj_datetime(), 'Adding size column to image_filenames dataframe...')
  train = im_stats(train)
  train = train[train['size'] != '0 0'].reset_index(drop=True) #remove bad images
  
  #quit() - Corregido con if __name__ ==  '__main__'
  
  print(jj_datetime(), 'Normalizing image features...({:})'.format(n_resize_to))
  train_data = normalize_image_features(train['path'], resize_to = n_resize_to)
  str_fichname = 'train' + jj_input_filename_suffix(n_resize_to, b_con_muestreo) + '.npy'  # jj_input_filename_suffix == '_{:}'.format(n_resize_to) + ('_prueba' if b_con_muestreo else '')
  np.save(s_input_path + str_fichname, train_data, allow_pickle=True, fix_imports=True)
  print(jj_datetime(), 'Ok train_data (' + str_fichname + ') saved:', train_data.shape)
  
  le = LabelEncoder()
  train_target = le.fit_transform(train['type'].values)
  print(le.classes_) #in case not 1 to 3 order
  str_fichname = 'train_target' + jj_input_filename_suffix(n_resize_to, b_con_muestreo) + '.npy'  # jj_input_filename_suffix == '_{:}'.format(n_resize_to) + ('_prueba' if b_con_muestreo else '')
  np.save(s_input_path + str_fichname, train_target, allow_pickle=True, fix_imports=True)
  print(jj_datetime(), 'Ok train_target (' + str_fichname + ') saved:', train_target.shape)
  
  test = glob.glob(s_input_path + 'test/*.jpg')
  test = [x.replace('\\', '/') for x in test] # Quitamos s_input_path y cambiamos \ por / [Windows!]
  test = pd.DataFrame([[p.split('/')[-1],p] for p in test], columns = ['image','path']) #[::20] #limit for Kaggle Demo
  print(jj_datetime(), test.shape)
  if b_con_muestreo:
    test = test[::20] #limit for Kaggle Demo (seleccionamos uno de cada 20 registros)
    print(jj_datetime(), 'After sampling:', test.shape)
  
  #quit() - Corregido con if __name__ ==  '__main__'
  
  print(jj_datetime(), 'Normalizing image features...({:})'.format(n_resize_to))
  test_data = normalize_image_features(test['path'], resize_to = n_resize_to)
  test_id = test.image.values
  str_fichname =  'test'    + jj_input_filename_suffix(n_resize_to, b_con_muestreo) + '.npy'  # jj_input_filename_suffix == '_{:}'.format(n_resize_to) + ('_prueba' if b_con_muestreo else '')
  str_fichname2 = 'test_id' + jj_input_filename_suffix(n_resize_to, b_con_muestreo) + '.npy'  # jj_input_filename_suffix == '_{:}'.format(n_resize_to) + ('_prueba' if b_con_muestreo else '')
  np.save(s_input_path + str_fichname,  test_data, allow_pickle=True, fix_imports=True)
  np.save(s_input_path + str_fichname2, test_id,   allow_pickle=True, fix_imports=True)
  print(jj_datetime(), 'Ok (test_data, test_id) (' + str_fichname + ', ' + str_fichname2 + ') saved:', (test_data.shape, test_id.shape))
  
  print(jj_datetime(), 'Ok.')

