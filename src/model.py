from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


def draw_history(history):
    # plt.plot(history.history['loss'], color='r')
    # plt.plot(history.history['val_loss'], color='g')
    plt.plot(history.history['rmse'], color='b')
    plt.plot(history.history['val_rmse'], color='k')
    plt.title('Model Acc and Loss')
    plt.ylabel('loss')
    # plt.yscale('log')
    plt.xlabel('epoch')
    # plt.legend(['train_loss', 'test_loss', 'train_rsme', 'test_rsme'], loc='upper left')
    plt.legend(['train_rsme', 'test_rsme'], loc='upper left')
    plt.savefig("rmse_loss.png")
    plt.show()


def light_model(x_train=None, y_train=None, x_test=None, y_test=None, is_train=False):
    model = Sequential()
    initializer = tf.keras.initializers.GlorotNormal()
    model.add(Dense(input_dim=x_train.shape[1], activation='selu', units=128, kernel_initializer=initializer))
    model.add(Dropout(0.2))
    # kernel_regularizer='l1'
    model.add(Dense(activation='selu', units=64, kernel_initializer=initializer, kernel_regularizer='l2'))
    model.add(Dense(activation='selu', units=32, kernel_initializer=initializer))
    model.add(Dense(units=1))
    
    opt = tf.keras.optimizers.RMSprop(learning_rate=4e-3, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,name='RMSprop')

    model.compile(loss="mse", optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
                    
    print(model.summary())
    
    if is_train:
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30)
        history = model.fit(x_train, y_train, batch_size=64, epochs=400, validation_split=0.2, callbacks=[callback])
        result = model.evaluate(x_test, y_test)
        print(result)
        draw_history(history)
        model.save("./models/baseline_model.h5")
    
    return model


def baseline_model(x_train=None, y_train=None, x_test=None, y_test=None, is_train=False):
    model = Sequential()
    model.add(Dense(input_dim=x_train.shape[1], activation='relu', units=128, kernel_initializer="random_normal"))
    model.add(Dropout(0.3))
    model.add(Dense(activation='selu', units=32, kernel_regularizer='l2'))
    model.add(Dense(activation='selu', units=16))
    model.add(Dense(activation='selu', units=16))
    model.add(Dense(activation='selu', units=4))
    model.add(Dense(units=1))
    

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=2e-3,decay_steps=10000,decay_rate=0.95)
    opt = tf.keras.optimizers.RMSprop(learning_rate=2e-3, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,name='RMSprop')

    model.compile(loss="mse", optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    print(model.summary())
    
    if is_train:
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=35)
        history = model.fit(x_train, y_train, batch_size=64, epochs=400, validation_split=0.2, callbacks=[callback])
        result = model.evaluate(x_test, y_test)
        print(result)
        draw_history(history)
        model.save("./models/baseline_model.h5")
    
    return model


class emb_model(tf.keras.Model):
    def __init__(self, x_train):
        super().__init__(self)
        self.emb1 = Dense(activation='selu', units=10)
        self.emb2 = Dense(activation='selu', units=3)
        self.emb3 = Dense(activation='selu', units=1)
        self.emb4 = Dense(activation='selu', units=3)
        self.num_emb = Dense(activation='selu', units=12)
        self.linear1 = Dense(activation='selu', units=128, kernel_initializer="random_normal")
        self.dropout = Dropout(0.3)
        self.linear2 = Dense(activation='selu', units=32)
        self.linear3 = Dense(activation='selu', units=16)
        self.linear4 = Dense(activation='selu', units=16)
        self.linear5 = Dense(activation='selu', units=4)
        self.linear6 = Dense(activation='selu', units=1)
    
    def call(self, inputs):
        emb4 = self.emb4(inputs[:, -11:])
        emb3 = self.emb3(inputs[:, -14:-11])
        emb2 = self.emb2(inputs[:, -26:-14])
        emb1 = self.emb1(inputs[:, 12:-26])
        num_emb = self.num_emb(inputs[:, 0:12])
        x = tf.concat([num_emb, emb4, emb3, emb2, emb1], 1)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.linear6(x)
        
        return x

def get_emb_model(x_train=None, y_train=None, x_test=None, y_test=None, is_train=False):
    model = emb_model(x_train)
    print(model)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=35)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=2e-3,decay_steps=10000,decay_rate=0.95)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, decay=0.)
    # opt = tf.keras.optimizers.RMSprop(learning_rate=2e-3, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,name='RMSprop')

    model.compile(loss="mse", optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    if is_train:
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=35)
        history = model.fit(x_train, y_train, batch_size=64, epochs=400, validation_split=0.2, callbacks=[callback])
        result = model.evaluate(x_test, y_test)
        print(result)
        draw_history(history)
        model.save_weights("./models/emb_model")
        print(model.summary())
    
    return model