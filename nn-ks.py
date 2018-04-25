#Imports to use Keras for CNN
#Load Keras
import keras
#Load MNIST from Keras in cloud
from keras.datasets import mnist
#Load the Sequential Keras NN model
from keras.models import Sequential
#Load the dropout and flatten libraries and fully connected layer
from keras.layers import Dense, Dropout, Flatten
#Load the pooling and convolutional layers
from keras.layers import Conv2D, MaxPooling2D
#Load Keras backend
from keras import backend as K

#Batch processed each step before updating weights, minibatch(batch < dataset && batch > 1)
batch_size = 256
#Number of classes in output(e.g. total number of potential solutions 0-9 = 10 potential solutions)
num_classes = 10
#Number of epochs to train for(epoch = 1 complete iteration over data)
epochs = 15

# input image dimensions
rows, columns = 28, 28

#Load the data, split between training and testing sets from MNIST(half of MNIST for training and half for testing)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#
if K.image_data_format() == 'channels_first':
    #Reshape x training data into 3D array based on original shape with 1 element for pixel, rows and columns for image
    x_train = x_train.reshape(x_train.shape[0], 1, rows, columns)
    #Reshape x testing data into 3D array based on original shape with 1 element for pixel, rows and columns for image
    x_test = x_test.reshape(x_test.shape[0], 1, rows, columns)
    #Defining shape as tuple list of integers necessary by the model layers
    input_shape = (1, rows, columns)
else:
    #Reshape x training data into 3D array based on original shape with 1 element for pixel, rows and columns for image
    x_train = x_train.reshape(x_train.shape[0], rows, columns, 1)
    #Reshape x training data into 3D array based on original shape with 1 element for pixel, rows and columns for image
    x_test = x_test.reshape(x_test.shape[0], rows, columns, 1)
    #Defining shape as tuple list of integers necessary by the model layers
    input_shape = (rows, columns, 1)

#Keras allows data type specification which can cause speedup over "loose" types in python : float32
#Casting x training dataset into defined size
x_train = x_train.astype('float32')
#Casting x testing dataset into defined size
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#Show reults of shape of x training data
print('x_train shape:', x_train.shape)
#Show training and testing samples shape
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#Training data
y_train = keras.utils.to_categorical(y_train, num_classes)
#Testing data
y_test = keras.utils.to_categorical(y_test, num_classes)

#Initialize and create a Sequential model, it is a linear stack of layers you can pass to contructor to build
model = Sequential()
#Convolution Layer 1, 2 dimensional layer, uses relu for activation, 32 nodes with filter of size 3x3
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
#Convolution Layer 2, 2 dimensional layer,  uses relu for activation, 32 nodes with filter of size 3x3
model.add(Conv2D(64, (3, 3), activation='relu'))
#Pooling Layer 1, 2 dimensional layer, uses max pooling algorithm, uses pool of size 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
#Dropout amount(Percentage of nodes to randomly deactivate)
model.add(Dropout(0.25))
#Flatten to vector
model.add(Flatten())
#Fully Connected Layer 1, uses 128 nodes and ReLu function for activation
model.add(Dense(128, activation='relu'))
#Dropout amount(Percentage of nodes to randomly deactivate)
model.add(Dropout(0.5))
#Fully Connected Layer 2, output layer which contains total number of outputs(classes) and softmax activation function
model.add(Dense(num_classes, activation='softmax'))

#Build model, use crossentropy for loss calculation and the Adadelta optimizer for optimizing processing
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

#Fit model to data over ___ epochs with batch size size of ___ and validate against x data and y labels
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
#Evaluate accuracy of model using x test data and y test labels
score = model.evaluate(x_test, y_test, verbose=0)

#Print results of loss and accuracy
print('Test loss:', score[0])
print('Test accuracy:', score[1])