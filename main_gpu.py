import numpy as np
import pandas as pd
import model
import hashlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from skimage.transform import  resize

import datetime
import time

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


max_timeStr = '2020-12-31 12:00:00'
maxTime = datetime.datetime.strptime(max_timeStr, '%Y-%m-%d %H:%M:%S')
max_date_value = int(time.mktime(maxTime.timetuple()))
print("MAX_DATE",max_date_value)


min_timeStr = '2018-01-02 10:30:00'
minTime = datetime.datetime.strptime(min_timeStr, '%Y-%m-%d %H:%M:%S')
min_date_value = int(time.mktime(minTime.timetuple()))
print("MIN_DATE",min_date_value)


# print(datas)
def data_resize(x_data):
    x_data = np.array(x_data).astype(np.float32)
    x_data = x_data.reshape(x_data.shape[0], 9, 9)

    temp_x = []
    for x in x_data:
        x = np.resize(x, (26, 26))
        x = np.pad(x, 1, mode='constant')
        temp_x.append(np.array(x).reshape(28, 28, 1))
        # temp_x.append(np.array(resize(x,(28,28))).reshape(28,28,1))
    #print(np.argmax(flat_sub_image))
    #print(np.argmax(flat_sub_image))
    print(np.array(temp_x).shape)
    x_datas = np.array(temp_x)
    return np.array(x_datas)

def zero_padding(datas, num):
    new_list = []
    for data in datas:
        data += (num - len(data)) * [0]
        new_list.append(data)
    return new_list

def date_processing(column,new_datas,datas):

    temp = []
    print("Date Column", column)
    

    for i, data in enumerate(new_datas[column]):
        
        if (data == '\\N'):
            data = min_date_value
        elif len(data) == 8 :
            data_time = datetime.datetime.strptime(data, '%Y%m%d').date()
            data = int(time.mktime(data_time.timetuple()))
        else:
            data_time = datetime.datetime.strptime(data, '%Y-%m-%d %H:%M:%S').date()
            data = int(time.mktime(data_time.timetuple()))
        temp.append(data)

    #print(temp[2])
    return temp

def data_preprocessing(datas):

    check_hist = []

    print("전체 Row 갯수 : ", len(datas))
    columns = datas.columns


    print("전체 Column 갯수 : ", len(columns))
    new_datas = {}
    scaler = MinMaxScaler()
    for c in columns:
        new_datas[c] = datas[c].astype(str)
        if len(new_datas[c]) > 0:
            temp = []
            # date value 날짜 처리
            for data in new_datas[c]:
                temp.append(int(hashlib.md5(bytes(data, encoding='utf8')).hexdigest()[:8], 16))
            temp.insert(0, 0)
            temp.append(9999999999)

            X_MinMax = scaler.fit_transform(np.array(temp).reshape(-1, 1))
            new_datas[c] = np.array(X_MinMax[1:len(X_MinMax) - 1]).reshape(-1)
            print(new_datas[c][1])

    return new_datas


path = './sales_sample.csv'
datas = pd.read_csv(path, encoding='utf-8', header=0)
print(datas)

new_datas = data_preprocessing(datas)

datas = pd.DataFrame(new_datas)
datas = datas.values.tolist()
print("데이터 Shape : ", np.array(datas).shape)

x = np.array(datas)#np.array(zero_padding(datas, 81))

print(np.array(x).shape)
x_train = x[:-100]
x_test = x[-100:]

x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0],14,1)


x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0],14,1)

print("x_train's shape : ",np.array(x_train).shape)
x_test = np.array(data_resize(x_test))
print("x_test's shape : ",np.array(x_test).shape)



dcgan = model.DCGAN(x_train=x_train,x_test=x_test)
model_name = 'DCGAN_mnist_model'
#Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')

train_steps = 230000
timer = model.ElapsedTimer()
dcgan.train(train_steps=train_steps,epoch=400,batch_size=255, save_interval=500,predict_interval=1000)
timer.elapsed_time()
# # if not os.path.isdir(save_dir):
# #     os.makedirs(save_dir)
# # dcgan.generator().save_weights(os.path.join(save_dir, 'generator'.format(train_steps)))
# # dcgan.discriminator_model().save_weights(os.path.join(save_dir, 'discriminator'.format(train_steps)))

