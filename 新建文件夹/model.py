from keras import layers
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, \
Activation, Dense
from keras.models import Model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import Input

def first_model(input_shape, num_classes):

    input_shape = Input(shape=(64, 64, 3))
    x = Conv2D(96, (3, 3), strides=2, padding='same', activation='relu')(input_shape)
    x = BatchNormalization()(x)
   
    x = Conv2D(256, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
   
    x = Conv2D(384, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
   
    x = Conv2D(384, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
   
    x = Conv2D(256, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    
    x = Dense(384, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(94, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(6, activation='softmax')(x)
    

    model = Model(input_shape, x)
    return model