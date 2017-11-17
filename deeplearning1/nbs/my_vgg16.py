import keras
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D, Convolution2D, ZeroPadding2D
from keras.layers import Dense, BatchNormalization
from keras.layers import Lambda, Flatten, Dropout
from keras import backend as K
from keras.preprocessing import image
import matplotlib
from matplotlib import pyplot as plt

K.set_image_dim_ordering('th')

import numpy as np

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
def vgg_preprocess(x):
    """
        Subtracts the mean RGB value, and transposes RGB to BGR.
        The mean RGB was computed on the image set used to train the VGG model.

        Args: 
            x: Image array (height x width x channels)
        Returns:
            Image array (height x width x transposed_channels)
    """
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr

class Vgg():
    def __init__(self, batch_norm=False):
        self.batch_norm = batch_norm
        self.model = self._build_model()
        
        if batch_norm:
            weights_file = "vgg16_bn.h5"
        else:
            weights_file = "vgg16.h5"
            
        file = get_file(weights_file, "http://files.fast.ai/models/" + weights_file, cache_subdir="models")
        self.model.load_weights(file)
    
    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
        """
            Takes the path to a directory, and generates batches of augmented/normalized data.
            Yields batches indefinitely, in an infinite loop.

            See Keras documentation: https://keras.io/preprocessing/image/
        """
        return gen.flow_from_directory(path, target_size=(224,224),
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)
    
    def fit(self, train_batches, valid_batches, nb_epoch=1, verbose=1):
            self.model.fit_generator(train_batches, samples_per_epoch=train_batches.nb_sample, nb_epoch=nb_epoch,
                                     verbose=verbose, validation_data=valid_batches,
                                     nb_val_samples=valid_batches.nb_sample)
    
    def finetune(self, batches):
        self.model.pop()
        for layer in self.model.layers: layer.trainable = False
            
        self.model.add(Dense(batches.nb_class, activation="softmax"))
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        
        classes = list(iter(batches.class_indices))
        for c in batches.class_indices:
            classes[batches.class_indices[c]] = c
        self.classes = classes
    
    def _add_conv_block(self, model, num_blocks, num_filters):
        for _ in range(num_blocks):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(num_filters, 3, 3, activation='relu'))
        
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        return model
    
    def _add_dense_block(self, model, num_blocks, num_neurons, dropout=0.5):
        for _ in range(num_blocks):
            model.add(Dense(num_neurons, activation="relu"))
            if self.batch_norm:
                model.add(BatchNormalization())
            model.add(Dropout(p=dropout))
        
        return model
        
    
    def _build_model(self):
        model = Sequential()
        model.add(Lambda(vgg_preprocess, input_shape=(3,224,224), output_shape=(3,224,224)))
        
        self._add_conv_block(model, num_blocks=2, num_filters=64)
        self._add_conv_block(model, num_blocks=2, num_filters=128)
        self._add_conv_block(model, num_blocks=3, num_filters=256)
        for _ in range(2):
            self._add_conv_block(model, num_blocks=3, num_filters=512)
        
        model.add(Flatten())
        
        self._add_dense_block(model, num_blocks=2, num_neurons=4096, dropout=0.5)
        model.add(Dense(1000, activation="softmax"))
        
        return model

if __name__ == "__main__":
    print("do it")