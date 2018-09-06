from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
# build CNN
my1stCNN = Sequential()
my1stCNN.add(Conv2D(256, (3,3), input_shape = (64,64,3), activation = 'relu'))
my1stCNN.add(MaxPooling2D(pool_size = (2,2)))

my1stCNN.add(BatchNormalization())


my1stCNN.add(Conv2D(256, (3,3), activation = 'relu'))
my1stCNN.add(MaxPooling2D(pool_size = (2,2)))

my1stCNN.add(BatchNormalization())

my1stCNN.add(Conv2D(256, (3,3), activation = 'relu'))
my1stCNN.add(MaxPooling2D(pool_size = (2,2)))

my1stCNN.add(BatchNormalization())


my1stCNN.add(Conv2D(256, (3,3), activation = 'relu'))
my1stCNN.add(MaxPooling2D(pool_size = (2,2)))

my1stCNN.add(BatchNormalization())

my1stCNN.add(Flatten())

my1stCNN.add(Dense(units = 64, activation = 'relu'))

my1stCNN.add(Dropout(0.2))


my1stCNN.add(Dense(units = 64, activation = 'relu'))

my1stCNN.add(Dropout(0.2))


my1stCNN.add(Dense(units = 64, activation = 'relu'))

my1stCNN.add(Dropout(0.2))

my1stCNN.add(Dense(units = 64, activation = 'relu'))

my1stCNN.add(Dropout(0.2))

my1stCNN.add(Dense(units = 64, activation = 'relu'))

my1stCNN.add(Dropout(0.2))


my1stCNN.add(Dense(units = 1, activation = 'sigmoid'))

print(my1stCNN.summary())

my1stCNN.compile(optimizer = 'Nadam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 10,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   )
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary') 
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 5,
                                            verbose = 1, factor=0.8, min_lr = 0.00001)

history = my1stCNN.fit_generator(training_set,
                                 steps_per_epoch = 40,
                                 nb_epoch = 200,
                                 validation_data = test_set,
                                 nb_val_samples = 2000,
                                 callbacks =[learning_rate_reduction] 
                       #use_multiprocessing = True, 
                       #workers = 4
                                 )
plt.plot()

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
