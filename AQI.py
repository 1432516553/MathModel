#coding:utf-8
import pandas as pd
import numpy as np

IAQI_CLASS = None
POLLUTANTS_CLASS = ["SO2", "NO2", "PM10", "PM2.5", "O3", "CO"]
QUALITY = {
    "优": [0, 50],
    "良": [51, 100],
    "轻度污染": [101, 150],
    "中度污染": [151, 200],
    "重度污染": [201, 300],
    "严重污染": [301, np.inf] # numpy.inf 表示最大值
}


def read_IAQI_CLASS(filename="./data/IAQI_CLASS.xlsx"):
    data = pd.read_excel(filename)
    # print("data:", data)
    global IAQI_CLASS
    IAQI_CLASS = data.to_numpy()[:,1:]
    print("IAQI_CLASS.shape:", IAQI_CLASS.shape)
    print(IAQI_CLASS)

    
def cal_AQI(data):
    data = data[0][2:]
    IAQI = np.zeros(6)  # 对应一条数据可以计算出 6 个IAQI值
    for i in range(6):
        C_P = data[i]
        for j in range(6):
            if IAQI_CLASS[i+1][j] - C_P > 0:            
                BP_Hi = IAQI_CLASS[i+1][j]
                BP_Lo = IAQI_CLASS[i+1][j-1]
                IAQI_Hi = IAQI_CLASS[0][j]
                IAQI_Lo = IAQI_CLASS[0][j-1]
                # print("C_P:", C_P)
                # print("BP_Hi:", BP_Hi)
                # print("BP_Lo:", BP_Lo)
                # print("IAQI_Hi:", IAQI_Hi)
                # print("IAQI_Lo:", IAQI_Lo)
                # print("\n")
                break
        IAQI[i] = int((IAQI_Hi - IAQI_Lo) / (BP_Hi - BP_Lo) * (C_P - BP_Lo) + IAQI_Lo)
    print("IAQI:", IAQI)
    AQI = np.max(IAQI)
    polltant = POLLUTANTS_CLASS[np.where(IAQI == AQI)[0][0]]
    quality = ""
    for key in list(QUALITY.keys()):
        if AQI > QUALITY[key][0] and AQI < QUALITY[key][1]:
            quality = key
            break
    return AQI, polltant, quality

def read_data(filename = None, sheetname = None, datetime=None):
    data = pd.read_excel(filename, sheetname)
    # print(data)
    # print(data[data['监测日期']=='2020-08-25'])
    print(data[data['监测日期'] == datetime])
    return data[data['监测日期'] == datetime].to_numpy()

if __name__ == '__main__':
    read_IAQI_CLASS()
    filename = './data/附件1 监测点A空气质量预报基础数据.xlsx'
    sheetname = '监测点A逐日污染物浓度实测数据'
    date_list = ['2020-08-25', '2020-08-26', '2020-08-27', '2020-08-28']
    for date in date_list:
        data = read_data(filename, sheetname, date)
        # print("data[0]:", data[0][3])
        if np.NaN in data:
            continue
        print("###################")
        print("date", date, "AQI:", cal_AQI(data))
