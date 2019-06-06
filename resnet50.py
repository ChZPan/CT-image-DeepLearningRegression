import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D, Input, Add, \
                         Activation, ZeroPadding2D, BatchNormalization, \
                         AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.optimizers import Adam

# Define loss function (tensor)
def Eucl_distance_tensor(y_pred, y_true):
    d = y_pred - y_true
    return tf.reduce_mean(tf.norm(d, axis=1))


def Eucl_distance(y_pred, y_true):
    d = y_pred - y_true
    return np.linalg.norm(d, axis=1).mean()


def idn_block(X, f, filters, stage, block):
    """
    Implementation of the identity block
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main 
         path
    filters -- python list of integers, defining the number of filters in the 
               CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in 
             the network
    block -- string/character, used to name the layers, depending on their 
             position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value to later add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), 
               padding = 'valid', name = conv_name_base + '2a', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1),
               padding = 'same', name = conv_name_base + '2b',
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), 
               padding = 'valid', name = conv_name_base + '2c', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2c')(X)

    # Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
   
    return X
    

def conv_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main 
         path
    filters -- python list of integers, defining the number of filters in the 
               CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in 
             the network
    block -- string/character, used to name the layers, depending on their 
             position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    # First component of main path 
    X = Conv2D(F1, (1,1), strides = (s,s), name = conv_name_base + '2a',
               padding = 'valid', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(F2, (f,f), strides = (1,1), name = conv_name_base + '2b',
               padding = 'same', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(F3, (1,1), strides = (1,1), name = conv_name_base + '2c',
               padding = 'valid', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(F3, (1,1), strides = (s,s), padding = 'valid',
                        name = conv_name_base + '1',
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X
    

def ResNet50(input_shape, lr_power=-3.0, lr_decay=0.0, 
             extra_layers=None, dropouts=None):
    """
    Implementation of the popular ResNet50 the following architecture:

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    
    # Zero-Padding
    X = ZeroPadding2D((1, 1))(X_input) # mod (3,3) -> (1,1)
    
    # Stage 1
    X = Conv2D(64, (3, 3), strides=(1, 1), name='conv1', # mod (7,7) -> (3,3); (2,2) -> (1,1)
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = conv_block(X, f=3, filters=[64, 64, 256], 
                            stage=2, block='a', s=1)
    X = idn_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = idn_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = conv_block(X, f=3, filters=[128,128,512],
                            stage=3, block='a', s=2) 
    X = idn_block(X, f=3, filters=[128,128,512], stage=3, block='b')
    X = idn_block(X, f=3, filters=[128,128,512], stage=3, block='c')
    X = idn_block(X, f=3, filters=[128,128,512], stage=3, block='d')

    # Stage 4
    X = conv_block(X, f=3, filters=[256,256,1024],
                            stage=4, block='a', s=2) 
    X = idn_block(X, f=3, filters=[256,256,1024], stage=4, block='b')
    X = idn_block(X, f=3, filters=[256,256,1024], stage=4, block='c')
    X = idn_block(X, f=3, filters=[256,256,1024], stage=4, block='d')
    X = idn_block(X, f=3, filters=[256,256,1024], stage=4, block='e')
    X = idn_block(X, f=3, filters=[256,256,1024], stage=4, block='f')

    # Stage 5
    X = conv_block(X, f=3, filters=[512,512,2048],
                            stage=5, block='a', s=2) 
    X = idn_block(X, f=3, filters=[512,512,2048], stage=5, block='b')
    X = idn_block(X, f=3, filters=[512,512,2048], stage=5, block='c')

    # AVGPOOL 
    X = AveragePooling2D((2,2), name='avg_pool')(X)
    
    # Flatten
    X = Flatten()(X)
    
    # Add extra dense layers
    if extra_layers is not None:
        assert len(extra_layers) == len(dropouts), \
                "Arguments do Not match in length: extra_layers, dropouts."
        for i, layer, dpout in (zip(range(len(extra_layers)), extra_layers, dropouts)):
            X = Dense(layer, name='fc_'+str(i)+'_'+str(layer), activation='relu',
                      kernel_initializer=glorot_uniform(seed=0))(X)
            X = Dropout(dpout, seed=0, name='dropout_'+str(i)+'_'+str(dpout))(X)

    # Output 
    X = Dense(2, name='fc_outputs', kernel_initializer=glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name = 'ResNet50')
    
    # Compile model
    learning_rate = 10.0**(lr_power)
    optim = Adam(lr=learning_rate, decay=lr_decay)
    model.compile(loss=Eucl_distance_tensor, optimizer=optim)
    
    return model