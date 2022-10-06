import numpy as np
import pandas as pd
import tensorflow as tf

from keras import backend as K
K._get_available_gpus()

import os
base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale = 1./255
)

test_datagen = ImageDataGenerator(rescale = 1./255)

batch_size = 32
shape = (224, 224)

# importing layers and models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, InputLayer, BatchNormalization, Activation
from tensorflow.keras.models import Model

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications import DenseNet121, DenseNet201, ResNet50, ResNet152

shape = (224, 224)
train_gen = datagen.flow_from_directory(train_dir, target_size = shape, batch_size = batch_size, class_mode = 'categorical', color_mode = 'rgb', shuffle=False)
valid_gen = test_datagen.flow_from_directory(validation_dir, target_size = shape, batch_size = batch_size, class_mode = 'categorical', color_mode = 'rgb', shuffle=False)
test_gen = test_datagen.flow_from_directory(test_dir, target_size = shape, batch_size = batch_size, class_mode = 'categorical', color_mode = 'rgb', shuffle=False)

# VGG16
base_model = VGG16(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')
for layer in base_model.layers:
    layer.trainable = False    
predictions = Flatten()(base_model.output)
model_VGG16 = Model(inputs=base_model.input, outputs=predictions)

# VGG19
model_VGG19 = VGG19(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')

# InceptionV3
model_InceptionV3 = InceptionV3(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')

# InceptionV4
model_InceptionV4 = InceptionResNetV2(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')

# DenseNet121
model_DenseNet121 = DenseNet121(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')

# DenseNet201
model_DenseNet201 = DenseNet201(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')

model_list = [model_VGG16]
dropout_list = [0.6]
acc_list = []

for dropout in dropout_list:
    base_model = model_VGG16
    for layer in base_model.layers:
        layer.trainable = False    
    predictions = Flatten()(base_model.output)
    model = Model(inputs=base_model.input, outputs=predictions)

    from keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 200)
    from keras.callbacks import ModelCheckpoint
    mc1 = ModelCheckpoint('best_modelACC.h5', monitor = 'val_categorical_accuracy', mode = 'max', verbose = 0, save_best_only = True)
    mc2 = ModelCheckpoint('best_modelLOSS.h5', monitor = 'val_loss', mode = 'min', verbose = 0, save_best_only = True)

    from tensorflow.keras.optimizers import Adam
    adam = Adam(learning_rate=0.00001)

    
    # getting X_train, y_train, X_test, y_test, X_valid, y_valid from the generators
    train_gen.reset()
    X_train, y_train = next(train_gen)
    X_train = model.predict(X_train)
    for i in range(len(train_gen)-1): #1st batch is already fetched before the for loop.
        img, label = next(train_gen)
        X_train = np.append(X_train, model.predict(np.array(img)), axis = 0)
        y_train = np.append(y_train, label, axis=0)
    print(X_train.shape, y_train.shape)
    train_gen.reset()

    test_gen.reset()
    X_test, y_test = next(test_gen)
    X_test = model.predict(X_test)
    for i in range(len(test_gen)-1): #1st batch is already fetched before the for loop.
        img, label = next(test_gen)
        X_test = np.append(X_test, model.predict(np.array(img)), axis = 0)
        y_test = np.append(y_test, label, axis=0)
    print(X_test.shape, y_test.shape)
    test_gen.reset()

    valid_gen.reset()
    X_valid, y_valid = next(valid_gen)
    X_valid = model.predict(X_valid)
    for i in range(len(valid_gen)-1): #1st batch is already fetched before the for loop.
        img, label = next(valid_gen)
        X_valid = np.append(X_valid, model.predict(np.array(img)), axis = 0)
        y_valid = np.append(y_valid, label, axis=0)
    print(X_valid.shape, y_valid.shape)
    valid_gen.reset()

    from sklearn import utils
    utils.shuffle(X_train, y_train)
    
    # initializer = tf.keras.initializers.HeNormal()
    model = tf.keras.Sequential()
    model.add(InputLayer(input_shape=(X_train.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(1000, activation=tf.keras.activations.relu))
    model.add(Dropout(dropout))
    model.add(Dense(1000, activation=tf.keras.activations.relu))
    model.add(Dropout(dropout))
    model.add(Dense(1000, activation=tf.keras.activations.relu))
    model.add(Dropout(dropout))
    model.add(Dense(1000, activation=tf.keras.activations.relu))
    model.add(Dropout(dropout))
    model.add(Dense(3, activation="softmax"))

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    epochs = 2000

    history = model.fit(
        x = X_train,
        y = y_train,
        batch_size = 32,
        epochs=epochs,
        verbose = 0,
        validation_data=(X_valid, y_valid),
        callbacks = [es,mc1,mc2]
        )
    
    l = []
    
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print('> %.3f' % (acc * 100.0))
    
    modelACC = tf.keras.models.load_model("best_modelACC.h5")
    _, acc = modelACC.evaluate(X_test, y_test, verbose=0)
    print('> %.3f' % (acc * 100.0))
    l.append(acc)

    modelLOSS = tf.keras.models.load_model("best_modelLOSS.h5")
    _, acc = modelLOSS.evaluate(X_test, y_test, verbose=0)
    print('> %.3f' % (acc * 100.0))
    l.append(acc)
    acc_list.append(l)

print(acc_list)

model = tf.keras.Sequential()
model.add(model_VGG16)
model.add(Flatten())
model.add(modelACC)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.save("best_model.h5")