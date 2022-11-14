path  = 'b:/Hacathon/2 task/'          # зададим путь к директорию со входными файлами

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

X = np.load (path + 'X_test.npy')

def sensor_treshold_filtr(array, tres = 250):
    """Функция производит пороговую фильтрацию массива. 
    Исключаются строки, где максимальное значение не превышает пороговое значение.
    Аргументы: array - входной массив,
               tres - пороговое значение.
    Возвращает отфильтрованный массив."""
    low_sensors_set = set()               # 
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j].max() <= tres:
                low_sensors_set.add(j)
    low_sens_list = list(low_sensors_set)
    array_out = np.delete(array, low_sens_list, 1)
    return array_out

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# Фильтруем значения датчиков
X = sensor_treshold_filtr(X)
# Подготовка данных для подачи их в НН
# Меняем местами ось времени и ось показаний датчиков
X_nn = X.swapaxes(1, 2)


model_loaded = tf.keras.models.load_model(path + 'score98703.h5')

from statistics import mode

def smooth_row(row, mode_value=7):
    """Сглаживание дребезга.

    Сглаживание дребезга в строках с использованием моды.

    Аргументы:
        row: строка с дребезгом.
        mode_value: количество значений, по которым происходит сглаживание,
            по умолчанию 7.
    
    Возвращает:
        smoothing_row: сглаженная строка.

    """
    smoothing_row = []
    for k in range(len(row)):
        smoothing_row.append(mode(row[k:(k + mode_value)]))
    len(smoothing_row)
    return smoothing_row

y_pred_nn = model_loaded.predict(X_nn).argmax(axis=-1)

y_pred_smooth = []
for row in y_pred_nn:
    y_pred_smooth.append(smooth_row(row))
y_pred_smooth = np.array(y_pred_smooth)
y = y_pred_smooth


y_flat = np.concatenate([arr for arr in y_pred_smooth])

print(y)
