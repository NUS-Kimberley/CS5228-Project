from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

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

def baseline_model(x_train=None, y_train=None, x_test=None, y_test=None):
    model = Sequential()
    model.add(Dense(input_dim=x_train.shape[1], activation='relu', units=128, kernel_initializer="random_normal"))
    # old version: model.add(Dropout(0.35))
    model.add(Dropout(0.3))
    model.add(Dense(activation='relu', units=32))
    
    model.add(Dense(activation='relu', units=16))
    model.add(Dense(activation='relu', units=16))
    # old version: model.add(Dense(activation='relu', units=4))
    model.add(Dense(activation='relu', units=4))
    model.add(Dense(units=1))
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=35)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=2e-3,decay_steps=10000,decay_rate=0.95) #old :1.4e-3
    opt = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,name='RMSprop')

    model.compile(loss="mse", optimizer=opt,
                    metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    history = model.fit(x_train, y_train, batch_size=64, epochs=400, validation_split=0.2, callbacks=[callback])
    print(model.summary())
    result = model.evaluate(x_test, y_test)
    print(result)
    draw_history(history)
    model.save("./models/baseline_model.h5")