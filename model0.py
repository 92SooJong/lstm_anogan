import os, sys
import numpy as np
import time
import pandas as pd
import tensorflow as tf
import random
import hashlib
import model0_predict

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Input, Reshape, Dense, Dropout, UpSampling2D, Conv2D, Flatten
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.utils import multi_gpu_model
from keras.layers import Bidirectional,LSTM,TimeDistributed

import matplotlib.pyplot as plt


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self, sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time))


class DCGAN(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1, x_train=[], x_test=[]):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1
        self.D = None
        self.G = None

        self.x_train = x_train  # input_data.read_data_sets("mnist",one_hot=True).train.images
        self.x_train = np.array(self.x_train).reshape(-1, 74, 1).astype(np.float32)
        self.x_test = x_test

    def discriminator_model(self):
        optimizer = RMSprop(lr=0.00012)

        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.2
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (74,1)


        self.D.add(Bidirectional(LSTM(128,return_sequences=True),input_shape=(74,1)))
        self.D.add(Bidirectional(LSTM(64,return_sequences=True)))
        self.D.add(Bidirectional(LSTM(32,return_sequences=True)))
        self.D.add(Bidirectional(LSTM(16,return_sequences=True)))
        self.D.add(Bidirectional(LSTM(8,return_sequences=True)))
        self.D.add(Dense(1))
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.D


    def generator_model(self):

        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.3
        depth = 64*4  # 64 + 64 + 64 + 64
        dim = 7
        # In: 100
        # Out: dim x dim x depth

        #self.G.add(Dense((10),input_shape=(74,1)))
        self.G.add(Bidirectional(LSTM(256,return_sequences=True),input_shape=(74,1)))
        self.G.add(Bidirectional(LSTM(128,return_sequences=True)))
        self.G.add(Bidirectional(LSTM(64,return_sequences=True)))
        self.G.add(Bidirectional(LSTM(32,return_sequences=True)))
        self.G.add(Bidirectional(LSTM(16,return_sequences=True)))

        #self.G.add(TimeDistributed(Flatten()))
        self.G.add(Dense(1))
        self.G.add(Activation('sigmoid'))
        #self.G.add(Reshape((74,1)))



        return self.G


    def generator_containing_discriminator(self, g, d):
        optimizer = RMSprop(lr=0.0001)

        d.trainable = False
        ganInput = Input(shape=(74,1))

        x = g(ganInput)
        ganOutput = d(x)
        gan = Model(inputs=ganInput, outputs=ganOutput)
        gan.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return gan

    ################################ 여기서부터 새롭게 추간된 내용################################################################
    def sum_of_residual(self, y_true, y_pred):
        return tf.reduce_sum(abs(y_true - y_pred))

    def feature_extractor(self):
        d = self.discriminator_model()
        d.load_weights('./saved_models/discriminator')
        intermidiate_model = Model(inputs=d.layers[0].input, outputs=d.layers[-2].output)
        intermidiate_model.compile(loss='binary_crossentropy', optimizer='adam')
        return intermidiate_model

    def anomaly_detector(self):

        g = self.generator_model()
        g.load_weights('./saved_models/generator')
        g.trainable = False
        intermidiate_model = self.feature_extractor()
        intermidiate_model.trainable = False

        aInput = Input(shape=(74,1))
        gInput = Dense((74))(aInput)
        G_out = g(gInput)
        D_out = intermidiate_model(G_out)
        model = Model(inputs=aInput, outputs=[G_out, D_out])
        model.compile(loss=self.sum_of_residual, loss_weights=[1.0, 0.0], optimizer='adam')
        print("####Anomaly Model####")
        model.summary()
        return model

    def compute_anomaly_score(self, model, x):
        z = np.random.uniform(0, 1, size=(1, 74,1))
        intermidiate_model = self.feature_extractor()
        d_x = intermidiate_model.predict(x)
        loss = model.fit(z, [x, d_x], epochs=500, verbose=0)
        similar_data, _ = model.predict(z)
        return loss.history['loss'][-1], similar_data

    def get_feature_extractor(self):
        intermidiate_model = self.feature_extractor()
        return intermidiate_model


    def train(
            self,
            train_steps=2000,
            epoch=200,
            batch_size=256,
            save_interval=0,
            predict_interval=1000):

        d = self.discriminator_model()
        print("#### discriminator ####")
        d.summary()
        g = self.generator_model()
        print("#### generator ####")
        g.summary()
        d_on_g = self.generator_containing_discriminator(g, d)
        d.trainable = True
        total_step = 0
        for e in range(epoch):
            for i in range(int(len(self.x_train)/batch_size)):
                total_step +=1
                noise = np.random.uniform(0, 1, size=[batch_size, 74,1])
                images_train = self.x_train[
                               i*batch_size:i*batch_size+batch_size,
                               :,
                               :,
                               ]

                images_fake = g.predict(noise, verbose=0)

                x = np.concatenate((images_train, images_fake))
                y = np.ones([2 * batch_size, 1])
                y[batch_size:, :] = 0

                g.trainable = False
                d_loss = d.train_on_batch(x, y)
                g.trainable = True

                noise = np.random.uniform(0, 1.0, size=[batch_size, 74,1])

                d.trainable = False
                y = np.ones([batch_size, 1])
                a_loss = d_on_g.train_on_batch(noise, y)
                d.trainable = True

                log_mesg = "EPOCH %d STEP %d: [D loss: %f, acc: %f]" % (e,i, d_loss[0], d_loss[1])
                log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])

                if i % 100 == 0:
                    print(log_mesg)
                if save_interval > 0:
                    if (total_step) % save_interval == 0:
                        save_dir = os.path.join(os.getcwd(), 'saved_models')
                        g.save_weights(os.path.join(save_dir, 'generator'.format(1)), True)
                        d.save_weights(os.path.join(save_dir, 'discriminator'.format(1)), True)
                        # model = self.anomaly_detector()
                        #
                        # idx = random.randint(0, len(self.x_test) - 1)
                        # ano_score, similar_img = self.compute_anomaly_score(model, self.x_test[idx].reshape(1, 28, 28, 1))
                        # print("anomaly score : " + str(ano_score))
                        #
                        #
                        # f = open('graph.txt', 'a')
                        # data = str(i) + ',' + str(ano_score) + '\n'
                        # f.write(data)
                        # f.close()
                if (total_step) % predict_interval == 0:
                    model = self.anomaly_detector()
                    path = '../SMUF_TRAIN.csv'
                    train_data, columns = model0_predict.get_test_datas(path)

                    path = '../SMUF_TRUE.csv'
                    true_data, columns = model0_predict.get_test_datas(path)

                    path = '../SMUF_FALSE.csv'
                    false_data, columns = model0_predict.get_test_datas(path)

                    intermidiate_model = self.get_feature_extractor()
                    data_num = 5

                    train_sum_score, train_score_list = model0_predict.score(train_data, intermidiate_model, model, columns)
                    print("TRAIN AVG_Score : {}\n".format(train_sum_score / data_num))
                    false_sum_score, false_score_list = model0_predict.score(false_data, intermidiate_model, model, columns)
                    print("FALSE AVG_Score : {}\n".format(false_sum_score / data_num))
                    true_sum_score, true_score_list = model0_predict.score(true_data, intermidiate_model, model, columns)
                    print("TRUE AVG_Score : {}".format(true_sum_score / data_num))
                    print("False/True 비율 : {}%".format((false_sum_score / true_sum_score) * 100))
