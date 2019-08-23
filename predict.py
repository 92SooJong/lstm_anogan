import model0
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import resize
import time
import datetime
import csv

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from pylab import figure, text, scatter, show
import hashlib

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


dcgan = model0.DCGAN()

max_timeStr = '2020-12-31 12:00:00'
maxTime = datetime.datetime.strptime(max_timeStr, '%Y-%m-%d %H:%M:%S')
max_date_value = int(time.mktime(maxTime.timetuple()))

min_timeStr = '2018-01-02 10:30:00'
minTime = datetime.datetime.strptime(min_timeStr, '%Y-%m-%d %H:%M:%S')
min_date_value = int(time.mktime(minTime.timetuple()))



model = dcgan.anomaly_detector()
print(model)
data_num = 20


def zero_padding(datas, num):
    new_list = []
    for data in datas:
        data += (num - len(data)) * [0]
        new_list.append(data)
    return new_list

def date_processing(column,new_datas):

    temp = []
    #print("Date Column", column)
    for i,data in enumerate(new_datas[column][:20]):
        if (data == '\\N'):
            data = max_date_value
        elif len(data) == 8 :
            data_time = datetime.datetime.strptime(data, '%Y%m%d').date()
            data = int(time.mktime(data_time.timetuple()))
        else:
            data_time = datetime.datetime.strptime(data, '%Y-%m-%d %H:%M:%S').date()
            data = int(time.mktime(data_time.timetuple()))
        temp.append(data)
    #print(temp[2])
    return temp

def get_test_datas(path):
    test_datas = pd.read_csv(path, encoding='utf-8', header=0)
    print("테스트 전체 데이터 수",len(test_datas))

    columns = test_datas.columns
    print("총 Column 갯수 : ",len(columns))
    new_test_data = {}
    scaler = MinMaxScaler()

    date_list = ['FRST_ENTR_DTTM', 'ENTR_CHNG_DTTM',
                 'STTS_CHNG_DT', 'ENTR_STTS_CHNG_DTTM',
                 'FRST_RQST_DT', 'SYS_CREATION_DATE',
                 'BDP_LOAD_DTTM', 'SYS_UPDATE_DATA',
                 'CUST_SBGN_HOPE_DT', 'SBGN_APNT_DT']

    bw_list = ['OPERATOR_ID', 'APPLICATION_ID',
               'BDP_LOAD_USER_ID', 'IK_HPNO',
               'IK_HLDR_CUST_NO', 'IK_RLUSR_CUST_NO']

    for c in columns:
            new_test_data[c] = test_datas[c].astype(str)
            temp = []
            if c in date_list:
                temp = date_processing(c, new_test_data)
                temp.insert(0, min_date_value)
                temp.append(max_date_value)
            else:
                for data in new_test_data[c][:20]:
                    temp.append(int(hashlib.md5(bytes(data, encoding='utf8')).hexdigest()[:8], 16))
                temp.insert(0, 0)
                temp.append(9999999999)
            X_MinMax = scaler.fit_transform(np.array(temp).reshape(-1, 1))
            new_test_data[c] = np.array(X_MinMax[1:len(X_MinMax) - 1]).reshape(-1)
            print("Column : ",c)
            print("data ",new_test_data[c][0])


    new_test_data = pd.DataFrame(new_test_data)
    new_test_data = new_test_data.values.tolist()
    print("테스트 데이터 Shape : ",np.array(new_test_data).shape)

    x = np.array(zero_padding(new_test_data,81))
    print("padding 후 데이터 Shape",np.array(x).shape)

    #데이터 float type casting
    x_false_test = np.array(x).astype(np.float32)
    x_false_test = x_false_test.reshape(x_false_test.shape[0], 9, 9)
    print("테스트 데이터 이미지화 shape ",np.array(x_false_test).shape)

    temp_x = []
    for x in x_false_test:
        x = resize(x, (26, 26))
        x = np.pad(x, 1, mode='constant')
        temp_x.append(np.array(x).reshape(28, 28, 1))
    x_false_test = np.array(temp_x)
    print("최종 입력 데이터 Shape : ",x_false_test.shape)
    return x_false_test,columns


path = '../SMUF_TRAIN.csv'
train_data,columns = get_test_datas(path)

path = '../SMUF_TRUE.csv'
true_data,columns = get_test_datas(path)


path = '../SMUF_FALSE.csv'
false_data,columns = get_test_datas(path)


intermidiate_model = dcgan.get_feature_extractor()




def score(test_datas, intermidiate_model, model, data_num,columns,type):
    sum_score = 0
    score_list = []
    th = 0.0

    index_list = [str(i) for i in range(0,len(test_datas[0].flatten()))]
    csvfile = open('./score_' + type + '.csv', 'w')
    writer = csv.DictWriter(csvfile, fieldnames=index_list)
    writer.writeheader()

    dict_temp = dict.fromkeys(index_list,[])
    for test_img in test_datas[:data_num]:

        z = np.random.uniform(0, 1, size=(1, 100))
        d_x = intermidiate_model.predict(test_img.reshape(1, 28, 28, 1))

        loss = model.fit(z, [test_img.reshape(1, 28, 28, 1), d_x], epochs=500, verbose=0)
        similar_data, _ = model.predict(z)


        sub_img = np.subtract(similar_data,test_img.reshape(1, 28, 28, 1))
        flat_sub_img = abs(sub_img.flatten())

        #th_sub_img = [x if x > th else 0 for x in flat_sub_img]

        g_score = 0
        for pix in flat_sub_img:
            if pix > th:
                g_score += pix

        for column, value in zip(index_list, flat_sub_img):
            dict_temp[column] = value
            # if value > th:
            #    print("[ columns : {} , value : {} ]".format(column, value))
        writer.writerow(dict_temp)



        sum_score += (loss.history['loss'][-1])
        score_list.append(loss.history['loss'][-1])


        print(loss.history['loss'][-1])#loss.history['loss'][-1])
    #print(dict_datas)
    #df_datas=pd.DataFrame.from_dict(dict_datas)
    #print(df_datas)
    #df_datas.to_csv('./Scroe_'+type+'.csv')
    return sum_score, score_list


print("샘플 데이터 수 : ", data_num)


false_sum_score, false_score_list = score(false_data, intermidiate_model, model, data_num,columns,'FALSE')
print("FALSE SUM_Score : ", false_sum_score)
print("FALSE AVG_Score : ", false_sum_score / data_num)

train_sum_score, train_score_list = score(train_data, intermidiate_model, model, data_num,columns,'TRAIN')
print("TRAIN SUM_Score : ", train_sum_score)
print("TRAIN AVG_Score : ", train_sum_score / data_num)

true_sum_score, true_score_list = score(true_data, intermidiate_model, model, data_num,columns,'TRUE')
print("TRUE SUM_Score : ", true_sum_score)
print("TRUE AVG_Score : ", true_sum_score / data_num)





style.use('fivethirtyeight')

fig = plt.figure(figsize=(17, 10))
ax1 = fig.add_subplot(1, 1, 1)



bound_score = train_score_list[int(95/100 * len(train_score_list))]
print("bound score : ",bound_score)


f = open('predict_graph.txt','w')
data = str(data_num)+'\n'
f.write(data)


for value in train_score_list:
      data = str(value)+'\n'
      f.write(data)

for value in true_score_list:
      data= str(value)+'\n'
      f.write(data)

for value in false_score_list:
      data = str(value)+'\n'
      f.write(data)



def animate(i):
    xs = [x for x in range(1, data_num + 1)]
    bd = []
    for _ in range(data_num):
        bd.append(bound_score)

    ax1.clear()
    ax1.plot(xs, train_score_list, marker='o', color='green')
    ax1.plot(xs, false_score_list, marker='o', color='black')
    ax1.plot(xs, true_score_list, marker='o', color='blue')
    ax1.plot(xs, bd, color='red')
    text(0.2, 1.0, 'GREEN=TRIN', ha='center', va='center', transform=ax1.transAxes)
    text(0.5, 1.0, 'BLACK=FALSE', ha='center', va='center', transform=ax1.transAxes)
    text(0.9, 1.0, 'BLUE=TRUE', ha='center', va='center', transform=ax1.transAxes)
    plt.ylabel("Anomaly Score")
    plt.xlabel("Number of Step")


ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
