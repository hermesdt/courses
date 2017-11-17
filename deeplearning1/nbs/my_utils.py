import dateutil
import numpy as np
from IPython.display import FileLink
import matplotlib.pyplot as plt
from keras import backend as K

def now_str():
    return dateutil.tz.datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def create_link(ids, predictions, submission_filename="data/submission{}.csv".format(now_str())):
    output = np.column_stack([ids, predictions])
    print(output.shape)
                
    np.savetxt(submission_filename, output, fmt='%d,%.5f', header="id,label", comments="")
    return FileLink(submission_filename)

def to_plot(img):
    if K.image_dim_ordering() == 'tf':
        return np.rollaxis(img, 0, 1).astype(np.uint8)
    else:
        return np.rollaxis(img, 0, 3).astype(np.uint8)

def plot(img):
    plt.imshow(to_plot(img))

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

def split_on_last_layer(model, layer_class):
    idx = [index for index, layer in enumerate(model.layers) if type(layer) is layer_class][-1]
    initial_layers = model.layers[:idx+1]
    final_layers = model.layers[idx+1:]
    
    return (initial_layers, final_layers)

rms_optimizer = RMSprop(lr=0.00001, rho=0.7)
def create_fc_model(input_shape, model_weights):
    model = Sequential([
        MaxPooling2D(input_shape=input_shape),
        Flatten(),
        Dense(4096, activation="relu"),
        Dropout(0),
        Dense(4096, activation="relu"),
        Dropout(0),
        Dense(2, activation="softmax")
    ])
    
    for layer, layer_weights in zip(model.layers, model_weights):
        layer.set_weights([w/2 for w in layer_weights])
    
    model.compile(rms_optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

from keras.layers import BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten
from keras.utils.data_utils import get_file
from keras.optimizers import Adam

def create_fc_bn_layers(input_shape, dropout=0):
    return [
        MaxPooling2D(input_shape=input_shape),
        Flatten(),
        Dense(4096, activation="relu"),
        BatchNormalization(),
        Dropout(p=dropout),
        Dense(4096, activation="relu"),
        BatchNormalization(),
        Dropout(p=dropout),
        Dense(1000, activation="softmax")
    ]
        

def create_fc_bn_model(input_shape, dropout=0):
    model = Sequential(create_fc_bn_layers(input_shape=input_shape, dropout=dropout))
    
    model.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])
    return model

import keras.utils.np_utils

def onehot(labels):
    return keras.utils.np_utils.to_categorical(labels)

import bcolz

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]