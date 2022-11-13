# На вход подается массив X 
                  # форматом (50, N) (датчики, таймстепы) в файле pkl
                  # data_name - имя файла данных
data_name = '___'
path_data = './'+data_name+'.pkl' # путь к файлу данных

# и обученная модель, model_name - имя файла модели
model_name = 'm1_best'              # модель для первого пилота
path_model = './'+model_name+'.h5'  # путь к файлу модели


# Предикт сохраняется в файле "pred_<<data_name>>.pkl"


# количество таймстепов в окне, на котором обучена модель
n_ticks = 16
# Множество активных датчиков
activ_sens_set_comm = {0, 33, 2, 36, 5, 38, 8, 12, 15, 17, 19, 21, 24, 27, 29, 30}

import numpy as np
import pandas as pd
import tensorflow as tf
from statistics import mode
import pickle

# Загрузка данных
with open(path_data, 'rb') as f:
    X_rd = pickle.load(f)
# Загрузка модели    
m = tf.keras.models.load_model(path_model)    

# Фильтрация входного массива 
X_sel = X_rd[list(activ_sens_set_comm), :]



def get_predict(X, m, n_tick):
    """
    Функция формирует предикт. 
    Аргументы:
    X - входной массив признаков форматом (N, N_sensors),
    m - обученная модель,
    n_tic - количество тиков в окне, которое поступает на вход модели.
    Возвращает: массив из значений форматом (N, 1) N.    
    """
    y_pred = np.zeros((1, 1))
    for i in range(X.shape[0]//n_tick):
        X_i = np.expand_dims(X[i*n_tick:(i+1)*n_tick], axis = 0)
        y_pred_i = m.predict(X_i).argmax(axis=-1)
        y_pred_i = mode(y_pred_i[0])
        y_pred= np.hstack((y_pred, np.full((1, n_tick),y_pred_i)))
    y_pred = np.delete(y_pred, 0, axis=1)
    # Дополним в конце список предиктов до полного соответствия массиву признаков
    last_pred = y_pred[0,-1]
    y_pred= np.hstack((y_pred, np.full((1, X.shape[0]%n_tick), last_pred)))
    return y_pred


predict = get_predict(X_sel, m, n_ticks)

pkl_filename = "pred_"+data_name+".pkl" 
with open(pkl_filename, 'wb') as file: 
    pickle.dump(predict, file)

