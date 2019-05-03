from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization

# build CNN
my1stCNN = Sequential()
my1stCNN.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation = 'relu'))
my1stCNN.add(MaxPooling2D(pool_size = (2,2)))

# my1stCNN.add(BatchNormalization())

my1stCNN.add(Flatten())

my1stCNN.add(Dense(units = 128, activation = 'relu'))

# my1stCNN.add(Dropout(0.2))

my1stCNN.add(Dense(units = 1, activation = 'sigmoid'))

print(my1stCNN.summary())

my1stCNN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# 
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
my1stCNN.fit_generator(training_set, 
                       steps_per_epoch = 40,
                       nb_epoch = 40,
                       validation_data = test_set,
                       nb_val_samples = 2000
                       #use_multiprocessing = True, 
                       #workers = 4
                       )

image_path = 'dataset/test_set/dogs/dog.4002.jpg'
img = image.load_img(image_path, target_size=(64,64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = my1stCNN.predict(img)
print('Predicted:',preds)


import numpy as np
import cv2
import keras
import os
#cv2这个库需要你下，这个网址有教程，我刚才按照这个下的https://blog.csdn.net/zstarwalker/article/details/72855781
sourceDir = r"D:/WorkShop/CNN final/dataset/test_set/cats"
# 路径要写全
# 循环对目录里的图片进行预测，并得出结果
for file in os.listdir(sourceDir):
# 图片路径
    try:
        imgPath = os.path.join(sourceDir, file)
# 读取图片
        x = cv2.imread(os.path.expanduser(imgPath))
# 图片预处理：
# 1.由于我的模型训练是将所有图片缩放为64x64，所以这里也对图片做同样操作
        x = cv2.resize(x,dsize=(64,64),interpolation=cv2.INTER_LINEAR)
        x = x.astype(float)
# 2.模型训练时图片的处理操作（此处根据自己模型的实际情况处理）
        x *= (1./255)
        x = np.array([x])

        # 开始预测
        result = my1stCNN.predict(x)
        # 打印结果
        print(file + "  -->  " + str(result))
    except Exception as e:
        print(str(e))

#这里的try except结构是为了防止读到已损坏的数据。数据就是以0.5为界，真假我也看不出来了，编一下吧。
#先假定大于0.5是真吧
        













