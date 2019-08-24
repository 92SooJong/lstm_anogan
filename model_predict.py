
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import resize
import hashlib
import datetime
import time


max_timeStr = '2020-12-31 12:00:00'
maxTime = datetime.datetime.strptime(max_timeStr, '%Y-%m-%d %H:%M:%S')
max_date_value = int(time.mktime(maxTime.timetuple()))

min_timeStr = '2018-01-02 10:30:00'
minTime = datetime.datetime.strptime(min_timeStr, '%Y-%m-%d %H:%M:%S')
min_date_value = int(time.mktime(minTime.timetuple()))

def zero_padding(datas, num):
    new_list = []
    for data in datas:
        data += (num - len(data)) * [0]
        new_list.append(data)
    return new_list

def date_processing(column,new_datas,df_index):

    temp = []
    for i, data in zip(df_index,new_datas[column]):
        #if column == 'FRST_RQST_DT':
        if (data == '\\N'):
            data = min_date_value
        elif len(data) == 8:
            data_time = datetime.datetime.strptime(data, '%Y%m%d').date()
            data = int(time.mktime(data_time.timetuple()))
        else:
            data_time = datetime.datetime.strptime(data, '%Y-%m-%d %H:%M:%S').date()
            data = int(time.mktime(data_time.timetuple()))
        temp.append(data)
    return temp

def get_test_datas(path):
    real_datas = pd.read_csv(path, encoding='utf-8', header=0)
    print("테스트 전체 데이터 수",len(real_datas))

    columns = real_datas.columns
    print("총 Column 갯수 : ",len(columns))

    test_datas=real_datas.sample(n=5)
    df_index = []
    for row in test_datas.index:
        df_index.append(row)

    print("인덱스 : ",df_index)
    #print("테스트 데이타스\n ",test_datas)


    new_test_data = {}
    scaler = MinMaxScaler()
    for c in columns:
        new_test_data[c] = test_datas[c].astype(str)
        temp = []
        for data in new_test_data[c]:
            temp.append(int(hashlib.md5(bytes(data, encoding='utf8')).hexdigest()[:8], 16))
        temp.insert(0, 0)
        temp.append(9999999999)

        X_MinMax = scaler.fit_transform(np.array(temp).reshape(-1, 1))
        new_test_data[c] = np.array(X_MinMax[1:len(X_MinMax) - 1]).reshape(-1)

    print("HASH완료")
    new_test_data = pd.DataFrame(new_test_data)
    new_test_data = new_test_data.values.tolist()
    print("테스트 데이터 Shape : ",np.array(new_test_data).shape)


    #데이터 float type casting
    x_test = np.array(new_test_data).astype(np.float32)#/ 255.
    # data WxH reshape
    x_test = x_test.reshape(x_test.shape[0], 14, 1)
    print("테스트 데이터 이미지화 shape ",np.array(x_test).shape)

    print("최종 입력 데이터 Shape : ",x_test.shape)
    return x_test,columns


def score(test_datas, intermidiate_model, model,columns):
    sum_score = 0
    score_list = []
    th = 0.1

    for test_img in test_datas:
        z = np.random.uniform(0, 1, size=(1, 14,1))
        d_x = intermidiate_model.predict(test_img.reshape(1, 14, 1))

        loss = model.fit(z, [test_img.reshape(1, 14, 1), d_x], epochs=500, verbose=0)
        similar_data, _ = model.predict(z)
        
        # 가장 큰 값을 가지는 열 index 
        sub_image = np.subtract(similar_data,test_img.reshape(1,14,1))


        flat_sub_img = abs(sub_image).flatten()
        #th_sub_img = [x if x > th else 0 for x in flat_sub_img]

        g_score = 0
        for pix in flat_sub_img:
            if pix > th:
                g_score += pix


        # for column, value in zip(columns, flat_sub_img):
        #     if value > th:
        #         print("[ columns : {} , value : {} ]".format(column, value))


        # anomaly score
        sum_score += g_score#(loss.history['loss'][-1])
        score_list.append(g_score) #(loss.history['loss'][-1])
        
        print("[G_Score : {} ]\n".format(g_score))

    return sum_score, score_list
