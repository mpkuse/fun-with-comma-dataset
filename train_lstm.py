import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense
import numpy as np
import cv2

from load_data import KerasBatchGenerator


def build_lstm_net(input_shape):
    #---
    #--- Create Net
    #---
    model = Sequential()
    model.add(ConvLSTM2D(filters=6, kernel_size=(3, 3), strides=2,
                       input_shape=input_shape,
                       padding='same', return_sequences=True ))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=6, kernel_size=(3, 3),strides=2,
                       padding='same', return_sequences=True))
    model.add(BatchNormalization())

    # model.add(ConvLSTM2D(filters=6, kernel_size=(3, 3),strides=2,
                    #    padding='same', return_sequences=True))
    # model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=6, kernel_size=(3, 3),strides=2,
                       padding='same', return_sequences=False))
    model.add(BatchNormalization())

    model.add( Flatten() )

    model.add( Dense(500, activation='relu') )
    model.add( Dense(70, activation='softmax') )

    return model

if __name__ == '__main__':

    #---
    #--- Data Generator
    #---
    gen = KerasBatchGenerator(COMMA_PATH='/Bulk_Data/cv_datasets/comma/comma-dataset/')
    a,b = gen[0]


    # a.shape = (16, 60, 160, 320, 3); batch_size, t_step, width, height, chnl 
    model = build_lstm_net( input_shape=(a.shape[1], a.shape[2], a.shape[3], a.shape[4]) )
    model.summary()
    keras.utils.plot_model( model, show_shapes=True )
    model.load_weights( 'model_speed.keras')

    #---
    #--- Fit
    #---
    # optimizer = keras.optimizers.RMSprop(lr=0.0005)
    optimizer = keras.optimizers.Adadelta(  )

    import tensorflow as tf
    tb = tf.keras.callbacks.TensorBoard( )

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'] )
    if True:

        model.fit_generator( gen, epochs=5, verbose=1, callbacks=[tb] )
        model.save('model_speed.keras')
    else:
        model.load_weights( 'model_speed.keras' )
