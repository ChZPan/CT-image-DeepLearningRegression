import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D


# Define loss function
def Eucl_distance_tensor(y_pred, y_true):
    d = y_pred - y_true
    return tf.reduce_mean(tf.norm(d, axis=1))
    
    
def SimpleCNN(INPUT_SHAPE):
    
    # Create model
    # Input -> ((Conv2D->relu) x 2 -> MaxPool2D) x 2 -> 
    # Flatten -> Dense x 2 -> Output
    
    model = Sequential()
    
    model.add(Conv2D(filters = 32, kernel_size = (5,5), 
                     padding = 'Same', activation ='relu', 
                     input_shape = INPUT_SHAPE))
    
    model.add(Conv2D(filters = 32, kernel_size = (5,5),
                     padding = 'Same', activation ='relu'))
    
    model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 64, kernel_size = (3,3),
                     padding = 'Same', activation ='relu'))
    
    model.add(Conv2D(filters = 64, kernel_size = (3,3),
                     padding = 'Same', activation ='relu'))
    
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    # model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512, activation = "relu"))
    model.add(Dense(64, activation = "relu"))
    # model.add(Dropout(0.25))
    model.add(Dense(2))
    
    # Compile model
    model.compile(loss=Eucl_distance_tensor, optimizer='adam')
    
    return model