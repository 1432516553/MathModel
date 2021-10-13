import os
import pandas as pd
import numpy as np
import math


import tensorflow as tf
import pandas as pd
import numpy as np
import util
import tf


file_path = './train_set'
eps = 1.


def parse_file(file_name, train_flag=True, save_h5py=True):
    data = pd.read_csv(os.path.join(file_name))
    del data['Cell Index']

    x_min = min(data['X'].min(), data['Cell X'][0])
    y_min = min(data['Y'].min(), data['Cell Y'][0])
    data['X'] -= x_min
    data['Cell X'] -= x_min
    data['Y'] -= y_min
    data['Cell Y'] -= y_min

    cell_x, cell_y = data['Cell X'][0], data['Cell Y'][0]

    data['abs_height'] = data['Building Height'] + data['Altitude']
    data['cell_abs_height'] = data['Height'] + data['Cell Building Height'] + data['Cell Altitude']
    cell_height = data['cell_abs_height '][0]
    data['delta_height'] = data['abs_height'] - data['cell_abs_height']

    # log height
    data['abs_height_log'] = np.log10(data['abs_height'] + eps)
    data['cell_abs_height_log'] = np.log10(data['cell_abs_height '])

    # angle to pie
    data['Electrical Downtilt'] = data['Electrical Downtilt'] / 180 * np.pi
    data['Mechanical Downtilt'] = data['Mechanical Downtilt'] / 180 * np.pi
    data['Azimuth'] = data['Azimuth'] / 180 * np.pi
    data['downtilt'] = data['Electrical Downtilt'] + data['Mechanical Downtilt']
    data['downtilt_pi_cos'] = np.cos(data['downtilt'])
    data['downtilt_pi_sin'] = np.sin(data['downtilt'])
    data['elec_down_cos'] = np.cos(data['Electrical Downtilt'])
    data['elec_down_sin'] = np.sin(data['Electrical Downtilt'])
    data['mech_down_cos'] = np.cos(data['Mechanical Downtilt'])
    data['mech_down_sin'] = np.sin(data['Mechanical Downtilt'])
    data['azimuth_cos'] = np.cos(data['Azimuth'])
    data['azimuth_sin'] = np.sin(data['Azimuth'])

    # distance
    data['dis_3d'] = np.sqrt((data['X'] - cell_x) ** 2 + (data['Y'] - cell_y)
                              ** 2 + (data['abs_height'] - cell_height) ** 2)
    data['dis_2d'] = np.sqrt((data['X'] - cell_x) ** 2 + (data['Y'] - cell_y) ** 2)

    # log distance
    data['dis_3d_log'] = np.log10(data['dis_3d'] + eps)
    data['dis_2d_log'] = np.log10(data['dis_2d'] + eps)

    # log frequency
    data['frequency_band_log'] = np.log10(data['Frequency Band'])
    cell_emission_angel = (2.0*np.pi - data['Azimuth']) + np.pi / 2.0
    p1_0, p1_1 = np.cos(cell_emission_angel), np.sin(cell_emission_angel)
    p2_0 = data['X'] - data['Cell X']
    p2_1 = data['Y'] - data['Cell Y']
    data['hori_belta'] = (p1_0 * p2_0 + p1_1 * p2_1) / np.sqrt(p2_0 ** 2 + p2_1 ** 2)
    data['hori_belta'] = data['hori_belta'].fillna(1)

    # belta
    p1_0, p1_1, p1_2 = p1_0, p1_1, - 1 * np.tan(data['downtilt'])
    p2_0, p2_1, p2_2 = p2_0, p2_1, data['delta_height']
    data['beta'] = (p1_0 * p2_0 + p1_1 * p2_1 + p1_2 * p2_2) / np.sqrt(p1_0 ** 2 + p1_1** 2 + p1_2 ** 2) / np.sqrt(p2_0 ** 2 + p2_1 **2 + p2_2 ** 2)
    data['beta'] = data['beta'].fillna(1)

    # delta x, delta y
    data['delta_x'] = p2_0
    data['delta_y'] = p2_1

    # obstacle
    obstacle_num = []
    worst_obstacle_height = []
    worst_obstacle_height_rate = []
    worst_obstacle_delta_d = []
    worst_obstacle_type = []
    worst_obstacle_x, worst_obstacle_y = [], []
    cell_x, cell_y, cell_height = data['Cell X'][0], data['Cell Y'][0], data['cell_abs_height'][0]
    for j in range(len(data)):
        x_0, y_0 = data['X'][j], data['Y'][j]
        abs_height_0 = data[' abs_height'][j]
        dx_0, dy_0 = x_0 - cell_x, y_0 - cell_y

        x_1, y_1 = data['X'], data['Y']
        dx_1, dy_1 = x_1 - cell_x, y_1 - cell_y

        the_cos = (dx_0 * dx_1 + dy_0 * dy_1) / np.sqrt(dx_0 ** 2 + dy_0 ** 2) / np.sqrt(dx_1 ** 2 + dy_1 ** 2)
        cos_simi = (the_cos >= 0.99)
        between = ((x_1 - x_0) * (x_1 - cell_x)) <= 0
        height = np.sqrt(dx_1 ** 2 + dy_1 ** 2) / np.sqrt(dx_0 ** 2 +
                                                          dy_0 ** 2) * (abs_height_0 - cell_height) + cell_height

        height = height.fillna(cell_height)

        too_height = data[' abs_height'] >= height
        too_height_rate = (data[' abs_height'] - height) / height

        num = np.sum(cos_simi & between & too_height)
        if num == 0:
            num = 1
            heightest = j
        else:
            heightest = too_height_rate.loc[cos_simi & between & too_height].argmax()
            obstacle_num.append(num)
    
    worst_obstacle_height.append(data['abs_height'][heightest])
    worst_obstacle_height_rate.append(too_height_rate[heightest])
    worst_obstacle_delta_d.append(np.sqrt(dx_1 ** 2 + dy_1 ** 2)[heightest])
    worst_obstacle_type.append(data[' Clutter Index'][heightest])
    worst_obstacle_x.append(x_1[heightest])
    worst_obstacle_y.append(y_1[heightest])
    data['obstacle_num'] = np.array(obstacle_num)
    data['worst_obstacle_height'] = np.array(height)
    data['worst_obstacle_height_rate'] = np.array(worst_obstacle_height_rate)
    data['worst_obstacle_delta_d'] = np.array(worst_obstacle_delta_d)
    data['worst_obstacle_type'] = np.array(worst_obstacle_type)
    data['worst_obstacle_x'] = np.array(worst_obstacle_x)
    data['worst_obstacle_y'] = np.array(worst_obstacle_y)

    for j in range(1, 21):
        if j in [1, 3, 4, 9, 19, 20]:
            continue
        data['Clutter Index_{}'.format(j)] = (data['Clutter Index'] == j).astype(int)
        del data['Clutter Index']

    for j in range(1, 21):
        if j in [1, 3, 4, 9, 19, 20]:
            continue
            data['index_{}_sum'.format(j)] = data['Clutter Index_{}'.format(j)].sum()

        data = data.fillna(0)
        data[data == float('inf')] = 0

        if save_h5py:
            h5 = pd.HDFStore('/home1/xie/shuxue/' + file_name.split('/')[-1][:-3] + 'h5', 'w')
            h5['data'] = data
            h5.close()

        if train_flag:
            target = data['RSRP']
            data = data.drop(['RSRP'], axis=1)
            return data, target
        else:
            return data