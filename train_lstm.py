import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense
import numpy as np
import cv2

from load_data import KerasBatchGenerator

#---
#--- Data Generator
#---
gen = KerasBatchGenerator(COMMA_PATH='/Bulk_Data/cv_datasets/comma/comma-dataset/')
a,b = gen[0]

#---
#--- Create Net
#---
model = Sequential()
model.add(ConvLSTM2D(filters=6, kernel_size=(3, 3), strides=2,
                   input_shape=(a.shape[1], a.shape[2], a.shape[3], a.shape[4]),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=6, kernel_size=(3, 3),strides=2,
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=6, kernel_size=(3, 3),strides=2,
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=6, kernel_size=(3, 3),strides=2,
                   padding='same', return_sequences=False))
model.add(BatchNormalization())

model.add( Flatten() )

model.add( Dense(500, activation='relu') )
model.add( Dense(70, activation='softmax') )


model.summary()
keras.utils.plot_model( model, show_shapes=True )


#---
#--- Fit
#---
# optimizer = keras.optimizers.RMSprop(lr=0.0005)
optimizer = keras.optimizers.Adadelta(  )

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'] )
if True:
    # model.load_weights( 'model_text.keras')
    model.fit_generator( gen, epochs=4, verbose=1 )
    model.save('model_speed.keras')
else:
    model.load_weights( 'model_speed.keras' )
