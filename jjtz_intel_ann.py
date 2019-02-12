#!/usr/bin/env python3
#Quiero utf8: áéíóú
# NOTA: UNICODE - UTF8 - También! Si no hay acentos, conversiones -> UNICODE/UTF8 a UTF8 (ed. Unicode)
# NOTA: Para borrar todas la variables: %reset
# NOTA: Arrancar desde /home/jjtoharia/ con:
# source kaggle/bin/activate
# [Python]  - python -i kaggle/IntelCervicalCancerScreening/jjtz_intel_ann.py   b_con_muestreo n_resize_to b_guardarDatos [b_leerModeloAnt]
# [pyspark] - spark-submit --driver-memory 4G kaggle/IntelCervicalCancerScreening/ann_entrenar_RDD.py   b_con_muestreo n_resize_to b_guardarDatos [b_leerModeloAnt]
# [Spark01-Python] python jjtz_intel_ann.py  0 256 1 >> /root/Dropbox/Musica/Tmp_Kaggle/IntelCervicalCancerScreening/jjtz_intel_ann_Spark01.log 2>&1 &
b_Spark = False

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LambdaCallback
from keras.layers.core import Dense, Dropout, Flatten, Activation, Reshape
#keras v1: from keras.layers.convolutional import Convolution2D, MaxPooling2D # , ZeroPadding2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import RMSprop, Nadam
from keras.preprocessing.image import ImageDataGenerator
# conda install scikit-learn
from sklearn.model_selection import train_test_split
from keras import backend as K
#keras v1: K.set_image_dim_ordering('th') # ==set_image_data_format('channels_first') en keras v2
K.set_image_data_format('channels_first')
K.set_floatx('float32')#('float64')

# import pandas as pd
from pandas import DataFrame as pd_DataFrame, read_csv as pd_read_csv
# import numpy as np
from numpy import sqrt as np_sqrt, arange as np_arange
from numpy import load as np_load
from numpy import array as np_array, ndarray as ndarray, delete as np_delete # ndarray es el tipo. np_array() es para crear ndarrays...
from numpy.random import seed as np_random_seed

from matplotlib.pyplot import plot as plt_plot, show as plt_show, legend as plt_legend, savefig as plt_savefig

from jjtz_intel_funciones import timefunc, jj_datetime, jj_datetime_filename, jj_input_filename_suffix, jj_safe_exec

# import os # os.listdir(path); os.remove(pathname); os.rename(old,new); os.path.isfile(pathname)
from os.path import isfile as os_path_isfile
from os.path import isdir as os_path_isdir
from os import chdir as os_chdir, listdir as os_listdir
from os import remove as os_remove
# from os import rename as os_rename
#from os import environ as os_environ # plat_uname()[1].lower() es el nombre del equipo (en minúsculas), solo en Windows
from platform import uname as plat_uname # plat_uname()[1].lower() es el nombre del equipo (en minúsculas)
#plat_uname() => uname_result(system='Linux', node='spark01', release='3.16.0-4-amd64', version='#1 SMP Debian 3.16.43-2 (2017-04-30)', machine='x86_64', processor='')
#plat_uname() => uname_result(system='Windows', node='JJTZAPATA-W10', release='10', version='10.0.10586', machine='AMD64', processor='Intel64 Family 6 Model 60 Stepping 3, GenuineIntel')
from filecmp import cmp as filecmp_cmp

from sys import argv

print(argv)
b_con_muestreo  =  (int(argv[1]) == 1) if(len(argv) >  1)  else True  # Hacer muestreo (reducción) de las imágenes
n_resize_to     =   int(argv[2])       if(len(argv) >  2)  else 32    # Tamaño (cuadrado) de las imágenes redimensionadas [256, 64 ó 32]
b_guardarDatos  =  (int(argv[3]) != 0) if(len(argv) >  3)  else False # Guardar datos al entrenar [entrenar_modelo()]
mis_iteraciones =   int(argv[4])       if(len(argv) >  4)  else -1    # Iteraciones (máx.) para entrenar (0 para no hace nada -sólo crear modelo-)
b_leerModeloAnt =  (int(argv[5]) != 0) if(len(argv) >  5)  else False # Leer Modelo ya entrenado (o no) y seguir entrenando (se creará otro s_unique_filename)
fichjson        =       argv[6]        if(len(argv) >  6)  else ''    # Fichero json con el modelo
fichhdf5        =       argv[7]        if(len(argv) >  7)  else ''    # Fichero hdf5 con los pesos del modelo
mi_iter_ini_ant =   int(argv[8])       if(len(argv) >  8)  else -1    # Iteración inicial
mis_iter_ant    =   int(argv[9])       if(len(argv) >  9)  else -1    # Iteraciones para entrenar (0 para generar ficheros submit sin entrenar)
b_BuscaBestHdf5 = (int(argv[10]) != 0) if(len(argv) > 10)  else True  # Buscar mejor hdf5 (model2)?

# NOTA: Para crear submit (y json y val.hdf5 y tst.hdf5), sin buscar mejor hdf5:
#       jjtz_intel_ann.py 0 32 1 0 1 fichjson fichhdf5 0 0 0

b_borrar_hdf5 = True # Borrar los hdf5 anteriores (solo queda el mejor de cada entrenamiento)
s_unique_filename = jj_datetime_filename() # Todos los ficheros generados tendrán esta marca única!
s_pc_name = plat_uname()[1].lower()
G_b_GPU = True if s_pc_name == 'asus-casa-w10' else False

print('Params: (b_con_muestreo, n_resize_to, b_guardarDatos, mis_iteraciones, b_leerModeloAnt)=', (b_con_muestreo, n_resize_to, b_guardarDatos, mis_iteraciones, b_leerModeloAnt), (s_unique_filename))
if b_leerModeloAnt:
  print('Params: (fichjson, fichhdf5, mi_iter_ini_ant, mis_iter_ant)=', (fichjson, fichhdf5, mi_iter_ini_ant, mis_iter_ant))

if(len(argv) < 4):
  print('\n\nERROR: Faltan params: [b_con_muestreo, n_resize_to, b_guardarDatos]\n\n')
  quit()

s_Proyecto = 'Kaggle_Intel-MobileODT'
s_input_path = 'C:/Servidor/JJTZ_Sync/' + s_Proyecto + '/'
if not os_path_isdir(s_input_path):
  s_input_path = 'C:/Servidor/JJTZ-Sync/' + s_Proyecto + '/'

if not os_path_isdir(s_input_path):
  s_input_path = '//jjtzapata-w10/C$/Servidor/JJTZ_Sync/' + s_Proyecto + '/'

if not os_path_isdir(s_input_path):
  s_input_path = '/home/jose/kaggle/' + s_Proyecto + '/'

if not os_path_isdir(s_input_path):
  print('\n\nERROR: s_input_path [' + s_input_path + '] NO encontrado!\n\n')
  quit()

s_output_path = 'C:/Personal/Dropbox/Musica/Tmp_Kaggle/IntelCervicalCancerScreening/Out/'
if not os_path_isdir(s_output_path):
  s_output_path = s_output_path.replace('C:/Personal/Dropbox', 'C:/Users/Adriana/Dropbox')

if not os_path_isdir(s_output_path):
  s_output_path = '/root/Dropbox/Musica/Tmp_Kaggle/IntelCervicalCancerScreening/Out/'

if not os_path_isdir(s_output_path):
  print('\n\nERROR: s_output_path [' + s_output_path + '] NO encontrado!\n\n')
  quit()

print(jj_datetime(), 'computer_name = ' + s_pc_name)
print(jj_datetime(), 'input_path = ' + s_input_path)
print(jj_datetime(), 'output_path = ' + s_output_path)

np_random_seed(43)


### ------------------------------
### LEER DATOS:
### ------------------------------
s_filename_suffix = jj_input_filename_suffix(n_resize_to, b_con_muestreo)  # jj_input_filename_suffix == '_{:}'.format(n_resize_to) + ('_prueba' if b_con_muestreo else '')

# No hace falta... os_chdir(s_input_path) # Necesario para leer los ficheros...

train_data = np_load(s_input_path + 'train' + s_filename_suffix + '.npy')
train_target = np_load(s_input_path + 'train_target' + s_filename_suffix + '.npy')

####################################################################################################
# https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data:
# UPDATED 5/22: We revisited these images and found that we need to update 3 labels:
# 
#     train/Type_2/80 - type 3
#     train/Type_3/968 - type 1
#     train/Type_3/1120 - type 1
# 
# https://www.kaggle.com/scottykwok/making-sense-out-of-some-difficult-samples:
#     Although these images are of low quality (especially 146) they are sufficient to make an assessment:
#       show('../input/train/Type_3/146.jpg')             [1178]
#       show('../input/train/Type_2/184.jpg')             [ 566]
# https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/30337:
#     Premature end of JPEG file: train/Type_1/1339.jpg (missing about 45% data) [  63]
# ########################################
# Cambios incluidos desde el 28/08/2017:
# ########################################
#     train/Type_2/80 - type 3
assert(train_target[919] == 1)
train_target[919] = 2
#     train/Type_3/968 - type 1
assert(train_target[1472] == 2)
train_target[1472] = 0
#     train/Type_3/1120 - type 1
assert(train_target[1074] == 2)
train_target[1074] = 0
# Eliminamos las imágenes 1178, 566 y 63 (en este orden para no modificar sus posiciones):
train_data   = np_delete(train_data,   1178, 0)
train_target = np_delete(train_target, 1178, 0)
train_data   = np_delete(train_data,    566, 0)
train_target = np_delete(train_target,  566, 0)
train_data   = np_delete(train_data,     63, 0)
train_target = np_delete(train_target,   63, 0)
####################################################################################################

# Train-Valid-Test: 60-20-20:
x_train,x_val_train,y_train,y_val_train = train_test_split(train_data,train_target,test_size=0.4, random_state=43)
x_val_train,x_tst_train,y_val_train,y_tst_train = train_test_split(x_val_train,y_val_train,test_size=0.5, random_state=43)

print(jj_datetime(), '(x_train,x_val_train,x_tst_train,y_train,y_val_train,y_tst_train):', tuple((x.shape for x in (x_train,x_val_train,x_tst_train,y_train,y_val_train,y_tst_train))))

@timefunc
def crear_modelo(mi_loss='sparse_categorical_crossentropy', mi_optimizador='adamax', mis_metrics=['accuracy'], img_size=32, conv_list=[[8,5], [16,3]], pool_list=[None, [3,2]], neurons_list=[10, 10, 10], dropout_list=[0.1, 0.1, 0.1], mi_batchsize=None, b_verbose=1):
    # 26/05/2017: Incluyo batchsize para poder usar stateful=True en RNN (p.ej. LSTM)...
    n_capas = len(neurons_list)
    n_convs = len(conv_list)
    n_pools = len(pool_list)
    padd_size = 1
    assert(len(dropout_list) == 3)
    
    img_size_evol = img_size # Para calcular el tamaño final tras las Convoluciones y Pooling.
    model = Sequential()
    #
    # Convoluciones y Pooling:
    # NOTA: Entre una Conv y otra, ponemos un ZeroPadding2D((1,1)), pero no entre Conv y MaxPool.
    # NOTA2: Desde 27/05/2017, ponemos ZeroPadding DELANTE de una Conv y no detrás...
    #
    for ix in range(0, max(n_convs, n_pools)):
      if n_convs > ix:
        if not conv_list[ix] is None:
          conv_nb_filter = conv_list[ix][0] # img_size / 8
          krnl_size = conv_list[ix][1] # if img_size==32 else (5 if img_size==64 else 7)
          if ix == 0:
            # NOTA2: Desde 27/05/2017, ponemos ZeroPadding DELANTE de una Conv y no detrás...
            padd_size = int(krnl_size / 2)
            model.add(ZeroPadding2D((padd_size,padd_size), input_shape=(3, img_size, img_size))) # ZeroPadding2D((1,1))
            img_size_evol += 2 * padd_size # Convolution
            if b_verbose > 0: print('ZPad - img_size_evol:', img_size_evol) # DEBUG
            #keras v1: model.add(Convolution2D(nb_filter=conv_nb_filter, nb_col=krnl_size, nb_row=krnl_size, activation='relu', dim_ordering='th', input_shape=(3, img_size, img_size))) #use input_shape=(3, 64, 64)
            model.add(Conv2D(filters=conv_nb_filter, kernel_size=(krnl_size, krnl_size), activation='relu', data_format='channels_first', input_shape=(3, img_size, img_size))) #use input_shape=(3, 64, 64)
          else:
            # NOTA2: Desde 27/05/2017, ponemos ZeroPadding DELANTE de una Conv y no detrás...
            padd_size = int(krnl_size / 2)
            model.add(ZeroPadding2D((padd_size,padd_size))) # ZeroPadding2D((1,1))
            img_size_evol += 2 * padd_size # Convolution
            if b_verbose > 0: print('ZPad - img_size_evol:', img_size_evol) # DEBUG
            # # NOTA: Entre una Conv y otra, ponemos un ZeroPadding2D((1,1)), pero no entre Conv y MaxPool:
            # if pool_list[ix-1] is None  and  not conv_list[ix-1] is None:
            #   model.add(ZeroPadding2D((padd_size,padd_size))) # ZeroPadding2D((1,1))
            #   img_size_evol += 2 * padd_size # Convolution
            #   if b_verbose > 0: print('ZPad - img_size_evol:', img_size_evol) # DEBUG
            #keras v1: model.add(Convolution2D(nb_filter=conv_nb_filter, nb_col=krnl_size, nb_row=krnl_size, activation='relu', dim_ordering='th'))
            model.add(Conv2D(filters=conv_nb_filter, kernel_size=(krnl_size, krnl_size), activation='relu', data_format='channels_first'))
          img_size_evol -= krnl_size - 1 # Convolution
          if b_verbose > 0: print('Conv - img_size_evol:', img_size_evol) # DEBUG
      
      if n_pools > ix:
        if not pool_list[ix] is None:
          maxpoolsize, maxpoolstrde = pool_list[ix] # [3, 2] == Overlapping pooling; [2, 2] == typical pooling
          #keras v1: model.add(MaxPooling2D(pool_size=(maxpoolsize, maxpoolsize), strides=(maxpoolstrde, maxpoolstrde), dim_ordering='th'))
          model.add(MaxPooling2D(pool_size=(maxpoolsize, maxpoolsize), strides=(maxpoolstrde, maxpoolstrde), data_format='channels_first'))
          img_size_evol = int(img_size_evol / maxpoolstrde) - (1 if img_size_evol % 2 == 0 and maxpoolsize == 3 else 0)  # MaxPooling2D
          if b_verbose > 0: print('Pool - img_size_evol:', img_size_evol)
    
    #krnl_size = 3 if img_size==32 else (5 if img_size==64 else 7)
    #maxpoolsize, maxpoolstrde = [3, 2] # [3, 2] == Overlapping pooling; [2, 2] == typical pooling
    #conv_nb_filter = 8
    ##keras v1: model.add(Convolution2D(nb_filter=conv_nb_filter, nb_col=krnl_size, nb_row=krnl_size, activation='relu', dim_ordering='th'))
    ##keras v1: model.add(MaxPooling2D(pool_size=(maxpoolsize, maxpoolsize), strides=(maxpoolstrde, maxpoolstrde), dim_ordering='th'))
    #img_size_evol -= krnl_size - 1 # Convolution
    #img_size_evol = int(img_size_evol / maxpoolstrde) - (1 if img_size_evol % 2 == 0 else 0)  # MaxPooling2D
    
    if not dropout_list[0] is None:
      model.add(Dropout(dropout_list[0])) # Dropout(0.1)
    
    if b_verbose > 0:
      model.summary()
      print('img_size_evol:', img_size_evol)
    
    #
    # Capas de neuronas: [seq_len = img_size_evol desde 22/05 20:28] [seq_len = 1 desde 22/05 18:53] [seq_len = conv_nb_filter again from 24/05]
    #
    seq_len = conv_nb_filter # = img_size_evol # = 1 # = conv_nb_filter # = 4 # 8 # 12
    num_cols = img_size_evol * img_size_evol # = conv_nb_filter * img_size_evol # = conv_nb_filter * img_size_evol * img_size_evol # = img_size_evol * img_size_evol # = 6 * 6 if img_size==32 else (12 * 12 if img_size==64 else 63 * 63)
    dropout_in, dropout_U = dropout_list[1:3] # = [0.1, 0.1]
    
    lstm_neuronas_ini, lstm_neuronas_mid1, lstm_neuronas_mid2 = [0, 0, 0]
    lstm_neuronas_fin = 3
    model.add(Reshape((-1, num_cols))) # Flatten la parte de la imagen...
    
    mi_LSTM = LSTM if G_b_LSTM else GRU # Probamos las neuronas GRU (Gated Recurrent Unit) que tienen una puerta menos que las LSTM (la "forget gate")
    mi_implementation = 2 if G_b_GPU else 1 # 1==Less memory (more but smaller matrix operations); 2==Better parallelization combining recurrent gates in same matrices. [1 y, sobre todo 2, mejor para GPU]
    if b_verbose > 0:
      if mi_implementation != 0 or b_verbose > 1:
        print('crear_modelo(): mi_implementation:', mi_implementation)
    
    if n_capas > 0:
      lstm_neuronas_ini = neurons_list[0]
      #keras v1: model.add(mi_LSTM(input_length=seq_len, input_dim=num_cols, output_dim=lstm_neuronas_ini, dropout_W=dropout_in, dropout_U=dropout_U, return_sequences=True, stateful=(not mi_batchsize is None), implementation=mi_implementation)) # , activation='relu')) # , stateful: True) # para "recordar" el estado de cada neurona entre batches (la iésima posición de cada seq. se "recuerda" entre cada batch). Nota: Y entonces hay que poner shuffle a False al entrenar!!!
      model.add(mi_LSTM(batch_input_shape=(mi_batchsize, seq_len, num_cols), units=lstm_neuronas_ini, dropout=dropout_in, recurrent_dropout=dropout_U, return_sequences=True, stateful=(not mi_batchsize is None), implementation=mi_implementation)) # , activation='relu')) # , stateful: True) # para "recordar" el estado de cada neurona entre batches (la iésima posición de cada seq. se "recuerda" entre cada batch). Nota: Y entonces hay que poner shuffle a False al entrenar!!!
      if n_capas == 2:
          lstm_neuronas_mid1 = neurons_list[1]
          #keras v1: model.add(mi_LSTM(output_dim=lstm_neuronas_mid1, dropout_W=dropout_in, dropout_U=dropout_U, return_sequences=True, stateful=(not mi_batchsize is None), implementation=mi_implementation)) # , activation='relu'))
          model.add(mi_LSTM(units=lstm_neuronas_mid1, dropout=dropout_in, recurrent_dropout=dropout_U, return_sequences=True, stateful=(not mi_batchsize is None), implementation=mi_implementation)) # , activation='relu'))
      elif n_capas == 3:
          lstm_neuronas_mid1, lstm_neuronas_mid2 = neurons_list[1:3] # 1,2
          #keras v1: model.add(mi_LSTM(output_dim=lstm_neuronas_mid1, dropout_W=dropout_in, dropout_U=dropout_U, return_sequences=True, stateful=(not mi_batchsize is None), implementation=mi_implementation)) # , activation='relu'))
          #keras v1: model.add(mi_LSTM(output_dim=lstm_neuronas_mid2, dropout_W=dropout_in, dropout_U=dropout_U, return_sequences=True, stateful=(not mi_batchsize is None), implementation=mi_implementation)) # , activation='relu'))
          model.add(mi_LSTM(units=lstm_neuronas_mid1, dropout=dropout_in, recurrent_dropout=dropout_U, return_sequences=True, stateful=(not mi_batchsize is None), implementation=mi_implementation)) # , activation='relu'))
          model.add(mi_LSTM(units=lstm_neuronas_mid2, dropout=dropout_in, recurrent_dropout=dropout_U, return_sequences=True, stateful=(not mi_batchsize is None), implementation=mi_implementation)) # , activation='relu'))
      
      # Capa de salida:
      #keras v1: model.add(mi_LSTM(output_dim=lstm_neuronas_fin, dropout_W=dropout_in, dropout_U=dropout_U, activation='sigmoid', return_sequences=False, stateful=(not mi_batchsize is None), implementation=mi_implementation))
      model.add(mi_LSTM(units=lstm_neuronas_fin, dropout=dropout_in, recurrent_dropout=dropout_U, activation='sigmoid', return_sequences=False, stateful=(not mi_batchsize is None), implementation=mi_implementation))
    else:
      #No hay capas LSTM, solo la de salida:
      # Capa de salida:
      #keras v1: model.add(mi_LSTM(input_length=seq_len, input_dim=num_cols, output_dim=lstm_neuronas_fin, dropout_W=dropout_in, dropout_U=dropout_U, activation='sigmoid', return_sequences=False, stateful=(not mi_batchsize is None), implementation=mi_implementation))
      model.add(mi_LSTM(batch_input_shape=(mi_batchsize, seq_len, num_cols), units=lstm_neuronas_fin, dropout=dropout_in, recurrent_dropout=dropout_U, activation='sigmoid', return_sequences=False, stateful=(not mi_batchsize is None), implementation=mi_implementation))
    
    model.compile(loss=mi_loss, optimizer=mi_optimizador, metrics=mis_metrics) #loss='binary_crossentropy' es log_loss para binarios
    if b_verbose > 0:
        model.summary()
    return(model)

#crear_modelo(mi_loss, mi_optimizador, mis_metrics, img_size=32, conv_list=[[8,5], [16,3]], pool_list=[None, [3,2]], neurons_list=[10, 10, 10], dropout_list=[0.2, 0.2, 0.2], mi_batchsize=None, b_verbose=1)

def descr_modelo(model, num_reg_train, batchsize, tipo_descr = 1):
  num_capas, lstm_neuronas_ini, lstm_neuronas_mid1, lstm_neuronas_mid2, lstm_neuronas_fin = [0, 0, 0, 0, 0]
  cnv_input_shape, cnv_descr, bInCnv = [0, '', False]
  for nObj in range(0, len(model.get_config())):
    oObjCls = model.get_config()[nObj]['class_name']
    oObj    = model.get_config()[nObj]['config']
    # DEBUG print(oObjCls) # DEBUG
    if   oObjCls in ('Convolution2D', 'Conv2D'): # 'Conv2D' para keras v2
      if nObj == 0:
        cnv_input_shape = oObj.get('batch_input_shape', [0,0])[-1] # (None, 3, 32, 32)[-1] == 32
        assert(cnv_input_shape > 0)
      cnv_descr += ('_cnv-' if not bInCnv else '-') + str(oObj.get('nb_filter', oObj.get('filters', 0)))
      kernel_size_rows = oObj.get('nb_row', oObj.get('kernel_size', [0,-1])[0]) # 'kernel_size' para keras v2
      kernel_size_cols = oObj.get('nb_col', oObj.get('kernel_size', [0,-1])[1]) # 'kernel_size' para keras v2
      cnv_descr += '_' + str(kernel_size_rows)
      assert(kernel_size_rows == kernel_size_cols) # Por si acaso
      bInCnv = True
    elif oObjCls == 'ZeroPadding2D':
      # No hacemos nada con esta, salvo que sea la capa de entrada:
      if nObj == 0:
        cnv_input_shape = oObj.get('batch_input_shape', [0,0])[-1] # (None, 3, 32, 32)[-1] == 32
        assert(cnv_input_shape > 0)
    elif oObjCls == 'MaxPooling2D':
      if nObj == 0:
        cnv_input_shape = oObj.get('batch_input_shape', [0,0])[-1] # (None, 3, 32, 32)[-1] == 32
        assert(cnv_input_shape > 0)
      cnv_descr += ('_poo-' if not bInCnv else '_p') + str(oObj.get('pool_size', [0,-1])[0]) # 3 o 2
      assert(oObj.get('pool_size', [0,-1])[0] == oObj.get('pool_size', [0,-1])[1]) # Por si acaso
      assert(oObj.get('strides', [0,-1])[0] == 2) # Por si acaso
      assert(oObj.get('strides', [0,-1])[0] == oObj.get('strides', [0,-1])[1]) # Por si acaso
    elif oObjCls == 'Dropout':
      cnv_descr += ('_drp-' if not bInCnv else '_d') + str(oObj.get('p', oObj.get('rate', -1))) # 0.1 # 'rate' para keras v2
    elif oObjCls == 'Reshape':
      pass
    elif oObjCls in ('LSTM', 'GRU'):
      bInCnv = False # Por si acaso
      if num_capas == 0:
        lstm_input_shape = oObj.get('batch_input_shape', [0,0,0]) # (None, 32, 225)
        num_cols = oObj.get('input_dim', lstm_input_shape[-1])   # 'batch_input_shape' para keras v2  # (None, 32, 225)[-1] == 225
        seq_len = oObj.get('input_length', lstm_input_shape[-2]) # 'batch_input_shape' para keras v2  # (None, 32, 225)[-2] == 32
        dropout_in = oObj.get('dropout_W', oObj.get('dropout', -9999))  # 'dropout' para keras v2
        dropout_U = oObj.get('dropout_U', oObj.get('recurrent_dropout', -9999)) # 'recurrent_dropout' para keras v2
        
        lstm_neuronas_ini = oObj.get('output_dim', oObj.get('units', 0))
      elif num_capas == 1:
        lstm_neuronas_fin = oObj.get('output_dim', oObj.get('units', 0))
      elif num_capas == 2:
        lstm_neuronas_mid1 = lstm_neuronas_fin
        lstm_neuronas_fin = oObj.get('output_dim', oObj.get('units', 0))
      elif num_capas == 3:
        lstm_neuronas_mid2 = lstm_neuronas_fin
        lstm_neuronas_fin = oObj.get('output_dim', oObj.get('units', 0))
      num_capas += 1
    else:
      print('NOTA: class_name NO procesada:', oObjCls)
      assert(False)
  
  if tipo_descr == 1:
    descr = 'bch-' + str(batchsize) + '_reg-' + str(num_reg_train) + '_col-' + str(num_cols)
    descr = descr + ('_in-' + str(cnv_input_shape) if cnv_input_shape != 0  else '')
    descr = descr + (cnv_descr if cnv_descr != ''  else '')
    descr = descr + ('_lstm-' if G_b_LSTM else '_gru-') + str(lstm_neuronas_ini) + '_' + str(lstm_neuronas_mid1) + '_' + str(lstm_neuronas_mid2) + '_' + str(lstm_neuronas_fin) + '_d' + str(dropout_in) + '_' + str(dropout_U) #  + '_di-' + str(dropout_in) + '_du-' + str(dropout_U)
    descr = descr + '_seq-' + str(seq_len)
  else: # if tipo_descr == 2:
    descr = '(BatchSize = ' + str(batchsize) + ')' + '. (Dropout_in = ' + str(dropout_in) + '. Dropout_U = ' + str(dropout_U) + ')'
    descr = descr + '. (SeqLen = ' + str(seq_len) + ')'
    descr = descr + (' (CNN = in-' + str(cnv_input_shape) if cnv_input_shape != 0  else '')
    descr = descr + (cnv_descr if cnv_descr != ''  else '') + ')'
    descr = descr + (' - (LSTM = ' if G_b_LSTM else ' - (GRU = ')   + str(lstm_neuronas_ini)
    descr = descr + (            ',' + str(lstm_neuronas_mid1) if num_capas >= 2 else '')
    descr = descr + (            ',' + str(lstm_neuronas_mid2) if num_capas >= 3 else '')
    descr = descr +              ',' + str(lstm_neuronas_fin) + ')'
    descr = descr + ' - ' + str(num_reg_train) + ' regs' + '/' + str(num_cols) + ' cols'
  return(descr)

#descr_modelo(model, num_reg_train=x_train.shape[0], batchsize=mi_batchsize, tipo_descr = 1)

def guardar_modelo_json(model, num_reg_train, pre_post, batchsize, s_unique_filename):
  fichname = 'modelo' + pre_post + '_' + descr_modelo(model, num_reg_train, batchsize) + '_' + s_unique_filename + '.json'
  # Max. 256 chars para fullpathname, así que cortamos, si hace falta, antes de s_unique_filename:
  if len(s_output_path + fichname) > 255:
    fichname = 'modelo' + pre_post + '_' + descr_modelo(model, num_reg_train, batchsize)
    n_quitar_chars = len(s_output_path + fichname) + len('_' + s_unique_filename + '.json') - 255
    fichname = fichname[ :-n_quitar_chars-4] + '...' + '_' + s_unique_filename + '.json'
  print('Guardando modelo (json) [' + fichname + ']...')
  with open(s_output_path + fichname, 'w') as json_file:
    json_file.write(model.to_json())

def guardar_pesos(model, num_reg_train, batchsize, pre_post, iteraciones, val_loss, s_unique_filename):
  fich = 'weights_' + descr_modelo(model, num_reg_train, batchsize) + ('__{epoch:02d}-' + pre_post + '_{val_loss:.4f}').format(epoch=iteraciones, val_loss=val_loss) + '_' + s_unique_filename + '.hdf5'
  try:
    model.save_weights(s_output_path + fich)
  except:
    fich = ''
  return(fich)

def leer_modelo_json(fichname = 'modelopost_bch-15_reg-888_col-36_in1-32_cnv-16_5_poo-2_cnv-32_3_poo-2_drp-0.1_lstm-9_0_3_di-0.2_du-0.2_seq-32_20170511_225755.json'):
  json_file = open(s_output_path + fichname, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  return(model_from_json(loaded_model_json))

def leer_pesos_modelo(model, mi_loss='sparse_categorical_crossentropy', mi_optimizador='adamax', mis_metrics=['accuracy'], fichero_pesos = 'weights_bch-15_reg-888_col-36_in1-32_cnv-16_5_poo-2_cnv-32_3_poo-2_drp-0.1_lstm-9_0_3_di-0.2_du-0.2_seq-32__300-val_0.7947_20170511_225755.hdf5'):
  # load weights
  if not fichero_pesos is None  and  not os_path_isfile(s_output_path + fichero_pesos):
    print('ERROR: Fichero [' + fichero_pesos + '] NO encontrado.')
  else:
    print('Compilando modelo...'  if fichero_pesos is None  else  ('Cargando [' + fichero_pesos + '] y compilando modelo...'))
    model.compile(loss=mi_loss, optimizer=mi_optimizador, metrics=mis_metrics)
    if not fichero_pesos is None:
      model.load_weights(s_output_path + fichero_pesos)
  return(model)

#model = leer_pesos_modelo(leer_modelo_json(), mi_loss, mi_optimizador, mis_metrics)

from glob import glob as glob_glob
def bestWeightsFile(pattern, b_borrar_hdf5 = False, s_unique_filename = '', bVerbose = 0, b_evaluar_pesos = False):
  bestFich , valLossAnt = ['' , '{val_loss:.4f}'.format(val_loss=9)]
  bestFich2 , valLossAnt2 = [bestFich , valLossAnt]
  nBorrados = 0
  sLista = glob_glob(s_output_path + pattern)
  for sFich in sLista:
    sufLen = len('_' + jj_datetime_filename() + '.hdf5')
    sTmp = sFich[ :-sufLen] # Quitamos '_' + jj_datetime_filename() + '.hdf5'
    sTmp = sTmp[-6: ]  # Nos quedamos con {val_loss:.4f}
    if sFich.endswith('tst_' + sTmp + sFich[-sufLen: ]):
      continue # El de evaluación en tst lo ignoramos
    if sFich.endswith('val_' + sTmp + sFich[-sufLen: ]):
      continue # El de evaluación en tst lo ignoramos
    sFich = sFich.replace('\\', '/').replace(s_output_path, '') # Sin '\' y sin el path!
    if bVerbose > 1:
      print([sTmp, sFich]) # DEBUG
    if sTmp < valLossAnt:
      valLossAnt = sTmp
      bestFich = sFich
    if sTmp < valLossAnt2 and sFich.endswith(s_unique_filename + '.hdf5'):
      valLossAnt2 = sTmp
      bestFich2 = sFich
    if b_evaluar_pesos:
      evaluarFicheroPesos(sFich) # lee y compila modelo 'prev' correspondiente y evalua en train, valid y test. print [loss, val_loss, tst_loss].
  
  if bVerbose > 0:
    print('bestWeightsFile():', [valLossAnt, bestFich], [valLossAnt2, bestFich2 if valLossAnt != valLossAnt2 else ''])
  
  if (b_borrar_hdf5 or bVerbose > 0) and len(sLista) != 0:
    # Borramos los que NO sean el mejor, pero solamente con este mismo s_unique_filename.
    # i.e. puede haber dos "best" hdf5.
    mi_lista = [[sFich[ :-sufLen][-6: ] , sFich.replace('\\', '/')] for sFich in sLista if sFich.endswith(s_unique_filename + '.hdf5')]
    mi_lista = [a for a in mi_lista if not a[1].endswith('tst_' + a[0] + '_' + s_unique_filename + '.hdf5')]
    mi_lista = [a for a in mi_lista if not a[1].endswith('val_' + a[0] + '_' + s_unique_filename + '.hdf5')]
    for fich_val in mi_lista:
      if fich_val[0] not in (valLossAnt, valLossAnt2): # Bastaría con valLossAnt2, pero por si acaso...
        nBorrados = nBorrados + 1 # Contamos los que se deberían borrar (o que se van a borrar)
        if bVerbose > 0:
          print(('Eliminamos' if b_borrar_hdf5 else 'Encontrado') + ' hdf5 (not best):', [nBorrados, fich_val[0], fich_val[1].replace(s_output_path, '')])
        if b_borrar_hdf5:
          os_remove(fich_val[1])
  
  return([valLossAnt, bestFich, valLossAnt2, bestFich2 if valLossAnt != valLossAnt2 else '' , nBorrados])

#bestWeightsFile('weights_' + descr_modelo(model, num_reg_train=x_train.shape[0], batchsize=mi_batchsize) + '__*.hdf5', False, s_unique_filename, bVerbose=1)

def evaluarFicheroPesos(fichhdf5, b_crear_submit = False, mi_loss='sparse_categorical_crossentropy', mi_optimizador='adamax', mis_metrics=['accuracy']):
  # lee y compila modelo 'prev' correspondiente y evalua en train, valid y test. print [loss, val_loss, tst_loss]
  fichhdf5 = fichhdf5.replace('\\', '/').replace(s_output_path, '').replace('.hdf5', '') + '.hdf5'
  mi_s_unique_fich = fichhdf5[-20:-5]
  mod = [a.replace('\\', '/').replace(s_output_path, '') for a in glob_glob(s_output_path + 'modeloprev_*' + mi_s_unique_fich + '.json')]
  if len(mod) == 0: # hdf5 sin modelo (?):
    print('ERROR: No se ha encontrado modelo para ' + fichhdf5)
    return([99, 99, 99, fichhdf5])
  elif len(mod) != 1: # Un modelo correspondiente y solamente uno!
    print('WARN: Encontrado más de un modelo para ' + fichhdf5)
  fichjson = mod[0]
  mi_batchsize = int(fichjson[4+fichjson.find('bch-'):fichjson.find('_', 1+fichjson.find('bch-'))]) # batchsize a partir del nombre del json!!!
  assert(os_path_isfile(s_output_path + fichjson))
  model = leer_pesos_modelo(leer_modelo_json(fichjson), mi_loss, mi_optimizador, mis_metrics, fichero_pesos = fichhdf5)
  assert(n_resize_to == model.get_config()[0]['config']['batch_input_shape'][-1]) # Asegurarse de que x_train.shape es la correcta! (32, 64 o 256)
  loss = evaluar_modelo(model, x_train, y_train, jj_datetime() + 'train: ', batchsize=mi_batchsize, verbose = 1)
  loss_val = evaluar_modelo(model, x_val_train, y_val_train, jj_datetime() + 'val_train: ', batchsize=mi_batchsize, verbose = 1)
  loss_tst = evaluar_modelo(model, x_tst_train, y_tst_train, jj_datetime() + 'tst_train: ', batchsize=mi_batchsize, verbose = 1)
  # Primero verificamos que no haya sido creado ya un submit (submit_64_0.8486_20170524_082059*.csv:
  b_con_muestreo = False ############# Como ya tengo el Test, forzamos siempre el bueno.
  s_filename_suffix = jj_input_filename_suffix(n_resize_to, b_con_muestreo)  # jj_input_filename_suffix == '_{:}'.format(n_resize_to) + ('_prueba' if b_con_muestreo else '')
  s_prev_subm = 'submit' + '*' + s_filename_suffix + '_' + '{:.4f}'.format(loss_tst) + '_' + mi_s_unique_fich + '*' + '.csv' # submit*_64_0.8285_20170524_125655*.csv
  if 0 != len(glob_glob(s_output_path + s_prev_subm)):
    s_filename_submit = glob_glob(s_output_path + s_prev_subm)[0].replace('\\', '/').replace(s_output_path, '') # Nos quedamos con el primero
  else:
    s_filename_submit = ''
  if b_crear_submit:
    s_unique_filename = jj_datetime_filename()
    print(jj_datetime(), 'Predecimos en el Test...')
    test_data = np_load(s_input_path + 'test' + s_filename_suffix + '.npy')
    test_id = np_load(s_input_path + 'test_id' + s_filename_suffix + '.npy')
    pred = model.predict_proba(test_data)
    df = pd_DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
    df.insert(0, 'image_name', test_id) # df['image_name'] = test_id
    # df = df.sort_values('image_name')
    s_filename_submit = 'submit' + s_filename_suffix + '_' + '{:.4f}'.format(loss_tst) + '_' + s_unique_filename + '.csv' # submit_32_0.85949_20170514_111127.csv
    print(jj_datetime(), 'Creamos submit.csv... [' + s_filename_submit + ']')
    # Guardamos ficheros (submit, hdf5 y modelo):
    df.to_csv(s_output_path + s_filename_submit, index=False)
    fichPesos = guardar_pesos(model, num_reg_train=x_train.shape[0], batchsize=mi_batchsize, pre_post='tst', iteraciones=mis_iteraciones, val_loss=loss_tst, s_unique_filename=s_unique_filename)
    guardar_modelo_json(model, num_reg_train=x_train.shape[0], pre_post='copy', batchsize=mi_batchsize, s_unique_filename=s_unique_filename) # Guardamos estructura también al final.
  return([loss, loss_val, loss_tst, s_filename_submit if s_filename_submit != '' else fichhdf5]) # s_filename_submit=='' si b_crear_submit==False

#rets = [evaluarFicheroPesos(a) for a in glob_glob(s_output_path + '*in-64*20170524_*.hdf5') if a[-27:-21] < '0.8000']

from datetime import datetime
def jjtz_valida_fecha(date_text, fmt = '%Y-%m-%d'):
  try:
    datetime.strptime(date_text, fmt)
  except ValueError:
    return(False)
  return(True)

def buscar_weights(n = 1, b_stop_after_delete = True, b_show_next = False, b_borrar_hdf5 = False, b_evaluar_pesos = False):
  # Si b_stop_on_delete == True, se detiene en cuanto elimina los hdf5 de algun modelo:
  assert( not (b_borrar_hdf5 and b_evaluar_pesos) ) # O bien borramos, o bien evaluamos (xor!)
  lista_s_uniques = sorted(set([a.replace('\\', '/').replace(s_output_path, '')[-20:-5] for a in glob_glob(s_output_path + '*.json')]))
  lista_s_uniques = [s_unique for s_unique in lista_s_uniques if jjtz_valida_fecha(s_unique, '%Y%m%d_%H%M%S')]
  print('modelos únicos:', len(lista_s_uniques))
  tmp = [0,0,0,0,0]
  n = n - 1
  mi_b_borrar_hdf5 = b_borrar_hdf5
  while n < len(lista_s_uniques):
    pattern, s_unique_filename = [['weights_*' + s_unique_filename + '*.hdf5',s_unique_filename] for s_unique_filename in lista_s_uniques][n]
    tmp=bestWeightsFile(pattern, mi_b_borrar_hdf5, s_unique_filename, bVerbose = 1, b_evaluar_pesos = b_evaluar_pesos)
    n = n + 1
    print([n, s_unique_filename, tmp[4]], '\n')
    if b_stop_after_delete and tmp[4]!=0 and not (b_show_next and mi_b_borrar_hdf5):
      break
    if b_stop_after_delete and tmp[4]!=0 and b_show_next and mi_b_borrar_hdf5:
      mi_b_borrar_hdf5 = False # Seguimos una vez after borrar pesos, pero el sgte NO lo borramos...
  return(n)

#n_ult = buscar_weights(n = 1, b_show_next = True, b_borrar_hdf5 = 0) # buscar_weights(b_borrar_hdf5 = True, b_stop_after_delete = False) para borrar todos sin piedad!

def optimizador_descr(mi_optimizador):
  if type(mi_optimizador).__name__ == 'str':
    mi_optimizador_descr = 'opt-' + mi_optimizador
  else:
    mi_optimizador_descr = 'opt-' + type(mi_optimizador).__name__ + '_lr-{:.5f}'.format(mi_optimizador.get_config().get('lr',0)) + '_dec-{:.5f}'.format(mi_optimizador.get_config().get('schedule_decay', mi_optimizador.get_config().get('decay',0)))
  return(mi_optimizador_descr)

### ------------------------------
### CREAR MODELO:
### ------------------------------
assert(n_resize_to in (32,64,256,512,1024))
G_b_LSTM = False # Probamos las neuronas GRU (Gated Recurrent Unit) que tienen una puerta menos que las LSTM (la "forget gate")
if n_resize_to == 32:
  convoluciones = [ [16,3], [16,3], [32,3] ] # [filters, krnlsize]               # 0.84 (32) [[16,5], [32,3]]
  maxpooling    = [ None  , [2,2] , [2,2]  ] # [poolsize, strides] [3,2] o [2,2] # 0.84 (32) [[2,2] , [2,2] ]
  neuronas      = [  9  ]                                                # 0.84 (32) [9]
  dropouts      = [   0.1  ,  0.2  ,  0.2    ]                           # 0.84 (32) [0.1  ,  0.2  ,  0.2]
  rotation_deg  = 0.2 # DataGenerator()
  zoom_rng      = 0.2 # DataGenerator()
  shift_rng     = 0.2 # DataGenerator()
  shear_rng     = 0.2 # DataGenerator()
  augm_mult     = 5 # fit_generator()
  mi_batchsize = 32
elif n_resize_to == 64:
  convoluciones = [  [8,3] , [8,3] , [16,3]   ]  # [filters, krnlsize]
  maxpooling    = [  None  , [2,2] , [2,2]    ]  # [poolsize, strides] [3,2] o [2,2]
  neuronas      = [  14   ]
  dropouts      = [  0.1  ,  0.2 ,  0.2   ]
  rotation_deg  = 5 # DataGenerator()
  zoom_rng      = 0.5 # DataGenerator()
  shift_rng     = 0.2 # DataGenerator()
  shear_rng     = 0.2 # DataGenerator()
  augm_mult     = 10 # fit_generator()
  mi_batchsize = 224 # = 16 * 14
elif n_resize_to == 256:
  convoluciones = [  [ 8,3], [ 8,3], [12,3], [12,3], [16,3], [24,3] ] # [filters, krnlsize]
  maxpooling    = [  [2,2] , [2,2] , [2,2] , [2,2] , [2,2] , [2,2]  ] # [poolsize, strides] [3,2] o [2,2]
  neuronas      = [  24 ,  8    ]
  dropouts      = [   0.2  ,  0.2  ,  0.2 ]
  rotation_deg  = 2  # DataGenerator()
  zoom_rng      = 0.4 # DataGenerator()
  shift_rng     = 0.2 # DataGenerator()
  shear_rng     = 0.2 # DataGenerator()
  augm_mult     = 1  # fit_generator()
  mi_batchsize = 32
elif n_resize_to == 512:
  convoluciones = [  [ 8,3], [ 8,3], [ 8,3], [12,3], [16,3], [24,3], [32,3] ] # [filters, krnlsize]
  maxpooling    = [  [2,2] , [2,2] , [2,2] , [2,2] , [2,2] , [2,2] , [2,2]  ] # [poolsize, strides] [3,2] o [2,2]
  neuronas      = [  32 ,  16    ]
  dropouts      = [   0.2  ,  0.2  ,  0.2 ]
  rotation_deg  = 2  # DataGenerator()
  zoom_rng      = 0.4 # DataGenerator()
  shift_rng     = 0.2 # DataGenerator()
  shear_rng     = 0.2 # DataGenerator()
  augm_mult     = 4  # fit_generator()
  mi_batchsize = 32

mi_loss, mi_optimizador, mis_metrics = [ 'sparse_categorical_crossentropy', 'adamax', ['accuracy'] ]
# 15-05-2017 19:30: Cambiamos adamax por Nadam (i.e. Adam RMSprop with Nesterov momentum)
#mi_optimizador = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) # This optimizer is usually a good choice for recurrent neural networks.
#mi_optimizador = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004) # Nadam is Adam RMSprop with Nesterov momentum.
mi_optimizador =  Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.008) # learning rate y/o decay modificados...

if b_leerModeloAnt == False:
  print(jj_datetime(), 'Creando modelo...')
  model  = crear_modelo(mi_loss, mi_optimizador, mis_metrics, img_size=n_resize_to, conv_list=convoluciones, pool_list=maxpooling, neurons_list=neuronas, dropout_list=dropouts, b_verbose = 1)
  model2 = crear_modelo(mi_loss, mi_optimizador, mis_metrics, img_size=n_resize_to, conv_list=convoluciones, pool_list=maxpooling, neurons_list=neuronas, dropout_list=dropouts, b_verbose = 0) # Para valid. final
  print('Input Data Size: ({imgsize:d}) [{total:,d} x {augm_mult:,d}] = ({prms_estim:,.1f})^2'.format(imgsize=n_resize_to, total=x_train.shape[0] * 3 * n_resize_to * n_resize_to, augm_mult=augm_mult, prms_estim=np_sqrt(x_train.shape[0] * 3 * n_resize_to * n_resize_to * augm_mult)))

print(jj_datetime(), 'Creando DataGenerator(rotation_deg, zoom_rng, shift_rng, shear_rng)...', (rotation_deg,zoom_rng, shift_rng, shear_rng))
datagen = ImageDataGenerator(
        rotation_range     = rotation_deg, # [0-180]
        zoom_range         = zoom_rng,     # [0-1]
        width_shift_range  = shift_rng,    # [0-1]
        height_shift_range = shift_rng,    # [0-1]
        shear_range        = shear_rng,    # [0-1]
        horizontal_flip    = True,
        fill_mode          ='nearest')     # , featurewise_center=True, etc. (https://keras.io/preprocessing/image/)
datagen.fit(train_data) # Only required if featurewise_center or featurewise_std_normalization or zca_whitening. (https://keras.io/preprocessing/image/)

#stop()

losstr, lossva = [ [] , [] ]
def mi_plot_losses(epoch, logs, fichname = ''):
  global losstr, lossva, mi_iter_ini
  #print('\n[epoch, logs]:', [epoch, logs])
  if epoch != mi_iter_ini:
    #if len(losstr) < epoch:
    #  # Hemos empezado en una iteración intermedia... Rellenamos con unos:
    #  losstr = [1] * epoch
    #  lossva = [1] * epoch
    plt_plot(np_arange(mi_iter_ini, epoch),losstr)
    plt_plot(np_arange(mi_iter_ini, epoch),lossva)
    plt_legend(['train', 'valid'], loc='upper left')
    if fichname == '' and epoch % 10 == 0:
      plt_show() # Pruebas...
    elif fichname != '':
      plt_savefig(s_output_path + fichname, bbox_inches='tight')
  losstr.append(logs['loss'])
  lossva.append(logs['val_loss'])

@timefunc
def entrenar_modelo(model, x_train, y_train, x_val_train, y_val_train, batchsize, iteraciones, mi_early_stop = 5, mi_shuffle = True, b_guardar = False, n_verbose = 1, b_guardar_fotos = False, s_pc_name = '', s_unique_filename = 'ERROR', iter_inicial = 0, augm_mult = 1):
  num_reg_train = x_train.shape[0]
  guardar_modelo_json(model, num_reg_train, 'prev', batchsize, (s_pc_name + '_' if s_pc_name!='' else '') + optimizador_descr(mi_optimizador) + '_' + s_unique_filename) # Guardamos estructura ANTES de empezar...
  print('Entrenando el modelo (Iter = ' + str(iteraciones) + '). ' + descr_modelo(model, num_reg_train, batchsize, tipo_descr = 2))
  np_random_seed(1234)
  if b_guardar:
    fichname = 'weights_' + descr_modelo(model, num_reg_train, batchsize) + '__{epoch:02d}-{val_loss:.4f}' + '_' + s_unique_filename + '.hdf5'
    fichlog =  'fichlog_' + descr_modelo(model, num_reg_train, batchsize) + '_' + s_unique_filename + '.csv'
    fichpng =  'losses_' + descr_modelo(model, num_reg_train, batchsize) + '_' + s_unique_filename + '.png'
    callbacks = [
      ModelCheckpoint(s_output_path + fichname, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0), # Guardamos pesos si hemos mejorado
      EarlyStopping(monitor='val_loss', patience=mi_early_stop, verbose=1), # Paramos si llevamos demasiado tiempo sin mejorar
      CSVLogger(s_output_path + fichlog, append=True), # Guardamos history (csv con 'acc','loss','val_acc','val_loss') tras cada iteración
      LambdaCallback(on_epoch_end=lambda epoch, logs: mi_plot_losses(epoch, logs, fichpng)) # Plot the losses history tras cada iteración
    ]
  else:
    callbacks = [
      EarlyStopping(monitor='val_loss', patience=mi_early_stop, verbose=1)
    ]  
  #model.fit(x_train, y_train, validation_data=(x_val_train, y_val_train), nb_epoch=iteraciones, batch_size=batchsize)
  #model.fit(x_train, y_train, validation_data=(x_val_train, y_val_train), nb_epoch=iteraciones, batch_size=batchsize, shuffle=mi_shuffle, callbacks=callbacks)
  model.fit_generator(datagen.flow(x_train,y_train, batch_size=batchsize, shuffle=mi_shuffle
                                   , save_to_dir = (s_input_path if b_guardar_fotos else ''), save_prefix = 'tmp_')
                      , epochs=iteraciones, initial_epoch=iter_inicial
                      , steps_per_epoch=int((augm_mult * len(x_train))/batchsize), verbose=n_verbose, validation_data=(x_val_train, y_val_train), callbacks=callbacks)
  ##keras v1: model.fit_generator(datagen.flow(x_train,y_train, batch_size=batchsize, shuffle=mi_shuffle
  #                                 , save_to_dir = (s_input_path if b_guardar_fotos else ''), save_prefix = 'tmp_')
  #                    , nb_epoch=iteraciones, initial_epoch=iter_inicial
  #                    , samples_per_epoch=augm_mult * len(x_train), verbose=n_verbose, validation_data=(x_val_train, y_val_train), callbacks=callbacks)

from numpy import vstack as np_vstack
def evaluar_modelo(model, x_data, y_data, txt, batchsize, verbose = 2):
  if verbose > 2:
    print('Evaluamos el modelo (' + txt + '):')
  scores = model.evaluate(x_data, y_data, verbose=0, batch_size=batchsize)
  probs = model.predict_proba(x_data, batch_size=batchsize)
  if verbose > 1:
    print(np_vstack((probs[0:5].T, y_data[0:5].T)).T)
  # preds = model.predict_classes(x_data, batch_size=batchsize)
  # print(np_vstack((preds[0:5].T, y_data[0:5].T)).T)
  if verbose > 0:
    print(txt + '[Loss, 1-Loss)] =', ['{:.4f}'.format(scores[0]), '{:.2f}%'.format(100-scores[0]*100)])
  return(scores[0])

### ------------------------------
### ENTRENAR Y GUARDAR MODELO:
### ------------------------------
#mi_batchsize = 15
mi_iter_ini = 0
if mis_iteraciones == -1:  mis_iteraciones = 600
mi_early_stop = 250

if b_leerModeloAnt == True:
  #fichjson = 'modelopost_bch-15_reg-888_col-36_in1-32_cnv-16_5_poo-2_cnv-32_3_poo-2_drp-0.1_lstm-9_0_3_di-0.2_du-0.2_seq-32_20170511_225755.json'
  #fichhdf5 = 'weights_bch-15_reg-888_col-36_in1-32_cnv-16_5_poo-2_cnv-32_3_poo-2_drp-0.1_lstm-9_0_3_di-0.2_du-0.2_seq-32__300-val_0.7947_20170511_225755.hdf5'
  if fichjson == '':
    fichjson = input('fichero modelo json = ? :')
  assert(os_path_isfile(s_output_path + fichjson)) # El fichero json es obligatorio!
  # Sacamos el batchsize del nombre del fichero json!!!
  mi_batchsize = int(fichjson[4+fichjson.find('bch-'):fichjson.find('_', 1+fichjson.find('bch-'))])
  if fichhdf5 == '':
    fichhdf5 = input('fichero pesos hdf5 = (None)? :')
  if fichhdf5 == '' or fichhdf5 == 'None':
    fichhdf5 = None # Solamente compilar el modelo
  else:
    assert(os_path_isfile(s_output_path + fichhdf5)) # Aseguramos que existe el fichero
  model = leer_pesos_modelo(leer_modelo_json(fichjson), mi_loss, mi_optimizador, mis_metrics, fichero_pesos = fichhdf5)
  assert(n_resize_to == model.get_config()[0]['config']['batch_input_shape'][-1]) # Asegurarse de que x_train.shape es la correcta! (32, 64 o 256)
  model.summary()
  model2 = leer_modelo_json(fichjson) # Sin pesos, para buscar bestWeightsFile()
  print('Input Data Size: ({imgsize:d}) [{total:,d} x {augm_mult:,d}] = ({prms_estim:,.1f})^2'.format(imgsize=n_resize_to, total=x_train.shape[0] * 3 * n_resize_to * n_resize_to, augm_mult=augm_mult, prms_estim=np_sqrt(x_train.shape[0] * 3 * n_resize_to * n_resize_to * augm_mult)))
  mi_iter_ini = jj_safe_exec(0 if fichhdf5 is None else mis_iteraciones, int, input('mi_iter_ini = ? (' + str(0 if fichhdf5 is None else mis_iteraciones) + '):'))   if mi_iter_ini_ant == -1  else mi_iter_ini_ant
  if mi_iter_ini == 0 and not fichhdf5 is None:
    # Sacamos iter_ini del nombre del fichero hdf5 (NOTA: si es xxx__00-xxx, dará error el assert() sgte, de todas formas...):
    mi_iter_ini = int(fichhdf5[2+fichhdf5.find('__'):fichhdf5.find('-', 1+fichhdf5.find('__'))])
  assert((mi_iter_ini == 0 and fichhdf5 is None) or (mi_iter_ini > 0 and not fichhdf5 is None))
  mis_iteraciones = jj_safe_exec(mis_iteraciones, int, input('mis_iter_ant = ? (' + str(mis_iteraciones) + '):'))   if mis_iter_ant == -1  else mis_iter_ant
  mis_iteraciones = mi_iter_ini + mis_iteraciones # Incremento las iteraciones

if mis_iteraciones > mi_iter_ini:
  print(jj_datetime(), 'Entrenando modelo...')
  entrenar_modelo(model, x_train, y_train, x_val_train, y_val_train, batchsize=mi_batchsize, iteraciones=mis_iteraciones, mi_early_stop=mi_early_stop, b_guardar=b_guardarDatos, b_guardar_fotos=False, s_pc_name=s_pc_name, s_unique_filename=s_unique_filename, iter_inicial = mi_iter_ini, augm_mult = augm_mult)
elif(mi_iter_ini == 0):
  if b_BuscaBestHdf5:
    if 1 != jj_safe_exec(0, int, input('Crear fichero submit = (0)? :')):
      print('\n\nWARN: mis_iteraciones == mi_iter_ini == 0. No hacemos nada más...\n\n')
      quit()

guardar_modelo_json(model, num_reg_train=x_train.shape[0], pre_post='post', batchsize=mi_batchsize, s_unique_filename=s_pc_name + '_' + optimizador_descr(mi_optimizador) + '_' + s_unique_filename) # Guardamos estructura también al final.

### ------------------------------
### EVALUAR Y GUARDAR MODELO:
### ------------------------------
print(jj_datetime(), 'Evaluamos en el Val_train...')
loss_val = evaluar_modelo(model, x_val_train, y_val_train, jj_datetime() + 'val_train: ', batchsize=mi_batchsize, verbose = 1)
fichPesos = guardar_pesos(model, num_reg_train=x_train.shape[0], batchsize=mi_batchsize, pre_post='val', iteraciones=mis_iteraciones, val_loss=loss_val, s_unique_filename=s_unique_filename)
print(jj_datetime(), 'Evaluamos en el Tst_train...')
loss_tst = evaluar_modelo(model, x_tst_train, y_tst_train, jj_datetime() + 'tst_train: ', batchsize=mi_batchsize)
fichPesos = guardar_pesos(model, num_reg_train=x_train.shape[0], batchsize=mi_batchsize, pre_post='tst', iteraciones=mis_iteraciones, val_loss=loss_tst, s_unique_filename=s_unique_filename)

if b_BuscaBestHdf5:
  #print(jj_datetime(), 'Evaluamos el MEJOR modelo para Val_train (en Val_train y en Tst_train)...')
  
  bWF_list = bestWeightsFile('weights_' + descr_modelo(model, num_reg_train=x_train.shape[0], batchsize=mi_batchsize) + '__*.hdf5', False, s_unique_filename, bVerbose = 1)
  bestValLoss, bestWeightsFichero, bestValLoss2, bestWeightsFichero2 = bWF_list[0:4] # Cuando bestWeightsFichero2 != '', es el mejor de los que tienen la misma fecha_hora (s_unique_filename)
  b_BuscaBestHdf5 = bestWeightsFichero != '' and bestWeightsFichero != fichPesos
  if bestWeightsFichero != '' and bestWeightsFichero != fichPesos: # No queremos repetir # and not filecmp_cmp(s_output_path + bestWeightsFichero, s_output_path + fichPesos, shallow=False): # No queremos repetir
    model2 = leer_pesos_modelo(model2, mi_loss, mi_optimizador, mis_metrics, fichero_pesos = bestWeightsFichero)
    epsilon = 1e-9
    if max([max(a) if not isinstance(a[0], (list,tuple,ndarray)) else
         max([max(b) if not isinstance(b[0], (list,tuple,ndarray)) else
           max([max(c) if not isinstance(c[0], (list,tuple,ndarray)) else
             max([max(d) for d in c]) for c in b]) for b in a]) for a in abs(np_array(model.get_weights()) - np_array(model2.get_weights()))]) > epsilon:
      print(jj_datetime(), 'Evaluamos en el Val_train [BestIter]...')
      loss_val2 = evaluar_modelo(model2, x_val_train, y_val_train, jj_datetime() + 'val_train[BestIter]: ', batchsize=mi_batchsize, verbose = 1)
      print(jj_datetime(), 'Evaluamos en el Tst_train [BestIter]...')
      loss_tst2 = evaluar_modelo(model2, x_tst_train, y_tst_train, jj_datetime() + 'tst_train[BestIter]: ', batchsize=mi_batchsize)
    else:
      print('IGUALES!', [bestWeightsFichero, fichPesos])
      bestWeightsFichero = fichPesos

### ------------------------------
### CREAR TEST SUBMIT:
### ------------------------------
print(jj_datetime(), 'Predecimos en el Test...')
b_con_muestreo = False ############# Como ya tengo el Test, forzamos siempre el bueno.
s_filename_suffix = jj_input_filename_suffix(n_resize_to, b_con_muestreo)  # jj_input_filename_suffix == '_{:}'.format(n_resize_to) + ('_prueba' if b_con_muestreo else '')

test_data = np_load(s_input_path + 'test' + s_filename_suffix + '.npy')
test_id = np_load(s_input_path + 'test_id' + s_filename_suffix + '.npy')

pred = model.predict_proba(test_data)
df = pd_DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
df.insert(0, 'image_name', test_id) # df['image_name'] = test_id
# df = df.sort_values('image_name')
s_filename_submit = 'submit' + s_filename_suffix + '_' + '{:.4f}'.format(loss_tst) + '_' + s_unique_filename + '.csv' # submit_32_0.85949_20170514_111127.csv
print(jj_datetime(), 'Creamos submit.csv... [' + s_filename_submit + ']')
df.to_csv(s_output_path + s_filename_submit, index=False)

if b_BuscaBestHdf5: # if bestWeightsFichero != '' and bestWeightsFichero != fichPesos:
  pred2 = model2.predict_proba(test_data)
  df2 = pd_DataFrame(pred2, columns=['Type_1','Type_2','Type_3'])
  df2.insert(0, 'image_name', test_id) # df2['image_name'] = test_id
  # df2 = df2.sort_values('image_name')
  #print('bestWeightsFichero:', [bestValLoss, bestWeightsFichero])
  s_filename_submit = 'submit2' + s_filename_suffix + '_' + '{:.4f}'.format(loss_tst2) + '_' + s_unique_filename + '.csv' # submit2_32_0.85949_20170514_111127.csv
  if df2.equals(df):
    print(jj_datetime(), 'No creamos submit2. Ya hemos utilizado el mejor hdf5!')
    pd_DataFrame().to_csv(s_output_path + s_filename_submit.replace('.csv', '') + '___REPETIDO.csv', index=False) # DataFrame empty!
  else:
    print(jj_datetime(), 'Creamos submit2.csv[BestIter]... [' + s_filename_submit + ']')
    df2.to_csv(s_output_path + s_filename_submit, index=False)

print(jj_datetime(), 'Ok.')

#Python -i jjtz_intel_ann.py 0 32 1 1 modelopost_bch-15_reg-888_col-25_in1-32_cnv-24_3_poo-3_cnv-48_3_poo-3_drp-0.2_lstm-12_0_0_3_di-0.2_du-0.2_seq-48_20170512_201217.json weights_bch-15_reg-888_col-25_in1-32_cnv-24_3_poo-3_cnv-48_3_poo-3_drp-0.2_lstm-12_0_0_3_di-0.2_du-0.2_seq-48__271-0.7732_20170512_201217.hdf5 300 0
