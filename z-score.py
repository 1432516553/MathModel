# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def smoothed_z_score_test(data):

    """  一点调参总结
    平滑z-score 测试
    :param y: 原DataFrame的数据列
    :param lag: 滞后数(初始滑动窗口大小) ， 建议设置为业务线的循环周期需要的天数, 看业务线的周期规律——估算出回归周期，乘上系数; 按天变化的设置为7*4天, 按周的设置为7*4*4天, 等
    :param threshold: 阈值 = 当前值超出前面所有的值的平均水平的绝对值 除以 前面所有的值的标准差的倍数 的上限， 建议2倍
    :param influence: 平滑系数，发生异常点时使用的平滑系数,(0,1)，值越大越受当前值的影响，及异常值的折算系数，建议0.5左右
    :return:
    """

    input_id = 157  # 输入数据的id
    output_id = 161  # 聚合并输出的数据的id

    start_date = '2017-01-01'  # 开始转换的时间
    end_date = '2020-01-31'  # 结束转换的时间

    y = data[input_id]['data_value']
    # 设置z-score参数
    lag, threshold, influence = 8 * 2, 2, 0.5

    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0] * len(y)
    stdFilter = [0] * len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i - 1]) > threshold * stdFilter[i - 1]:
            if y[i] > avgFilter[i - 1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i - 1]
            avgFilter[i] = np.mean(filteredY[(i - lag + 1):i + 1])
            stdFilter[i] = np.std(filteredY[(i - lag + 1):i + 1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i - lag + 1):i + 1])
            stdFilter[i] = np.std(filteredY[(i - lag + 1):i + 1])

    series_dict = dict(signals=np.asarray(signals),
                       avgFilter=np.asarray(avgFilter),
                       stdFilter=np.asarray(stdFilter)
                       )

    data[output_id] = data[input_id].copy()
    data[output_id]['data_value'] = np.asarray(series_dict['signals'])


def paint(dfs=[], labels=[], title='暂无'):
    assert len(dfs) == len(labels)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(16, 8))
    for i in range(0, len(dfs)):
        plt.plot(dfs[i]['data_value'], label=labels[i])
    plt.legend(loc='best')
    plt.title(title)
    plt.show()


def read_data(filename = None, sheetname = None):
    data = pd.read_excel(filename, sheetname, names=None)
    
    data2 = data.values.tolist()
    result = []
    for data3 in data2:
        result.append(data3[2])
    # print(result)
    return result[0:90]


if __name__ == '__main__':
    idx = pd.date_range('2019-04-16', periods=90, freq='D')
    filename = './data/附件1 监测点A空气质量预报基础数据.xlsx'
    sheetname = '监测点A逐日污染物浓度实测数据'
    data = read_data(filename, sheetname)
    data_value = pd.Series(data, index=idx)
    print(data)
    # 设置z-score参数

    df = pd.DataFrame({
        'data_time': idx,  # 时间列
        'data_value': data_value  # 数据列
    })

    data = {}
    data[157] = df
    data[161] = pd.DataFrame()
    smoothed_z_score_test(data)

    paint(dfs=[data[157], data[161]], labels=['origin', 'signal'])

