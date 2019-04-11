
#%%
#导入特征向量（知名网络结构）
from keras.models import *
from keras.layers import *
from keras.applications.inception_v3 import InceptionV3
#from keras.applications.mobilenet import Mobilenet
from keras.applications import *
from keras.applications.resnet50 import ResNet50
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import MobileNet
from keras.applications import imagenet_utils
from keras.applications.xception import Xception
from keras.preprocessing.image import *
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
import h5py
global ww
def write_gap(MODEL, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    print(x)
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    #使用预训练模型,加载预训练权重
    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory("F:/python_workspace/bingli/train3/train3/", image_size, shuffle=False, 
                                              batch_size=16)
    test_generator = gen.flow_from_directory("F:/python_workspace/bingli/test3", image_size, shuffle=False, 
                                             batch_size=16, class_mode=None)
    #加载训练数据
    train = model.predict_generator(train_generator,train_generator.samples//16+1,verbose=1)
    test = model.predict_generator(test_generator, test_generator.samples//16+1,verbose=1)
#    ww=model.get_weights()
#    print(len(ww))
#    print(ww)
    #预测得到特征向量
    with h5py.File("F:/python_workspace/bingli/gap/VGG16.h5") as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)


#write_gap(ResNet50, (224, 224))
#write_gap(MobileNet,(224,224,3))
#write_gap(InceptionResNetV2,(299,299))
#write_gap(InceptionV3, (299, 299))
#write_gap(Xception, (299, 299))
#write_gap(VGG16, (224, 224))
#write_gap(VGG19, (224, 224))
#h=h5py.File('./keras/gap/InceptionV3.h5', 'r')
#print(h['train'].shape)


#%%
#读取模型特征向量训练以及预测
import keras
from keras.models import Sequential
from keras.layers import Activation,Dropout,Dense
from keras.regularizers import l2
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
X_train = []
X_test = []
# "gap/ResNet50.h5" , "gap/Xception.h5","gap/VGG19.h5","gap/inceptionV3.h5", "gap/VGG16.h5"
filePath="F:/python_workspace/bingli/"
model_list=["gap/inceptionV3.h5","gap/VGG16.h5"]
for filename in model_list:
    filename=filePath+filename
    
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])
#print(X_train)
X_train = np.concatenate(X_train, axis=1)#拼接特征向量
X_test = np.concatenate(X_test, axis=1)
X_train, y_train = shuffle(X_train, y_train) #洗牌
#y_train= keras.utils.to_categorical(y_train) #单分类
#print(y_train)
y_train= keras.utils.to_categorical(y_train, num_classes=2) #多分类
#print(y_train)
#print(set(y_train)) #显示类别
#%%
#定制模型
model=Sequential()  #通过input_dim和input_shape来指定输入的维数
model.add(Dense(1024,input_dim=X_train.shape[1]))
#model.add(Dense(1024))
model.add(Dense(1024,activation='relu',W_regularizer=l2(0.01)))
#model.add(Dropout(0.5,input_shape=X_train.shape[1:],name="sa"))
#dout1=model.get_weights()
#print(dout1)
#model.add(Dense(500,activation='relu',W_regularizer=l2(0.01),use_bias=True))
#model.add(Dropout(0.5))
model.add(Dense(200,activation='relu',W_regularizer=l2(0.01)))
#model.add(Dropout(0.5,input_shape=X_train.shape[1:]))
#model.add(Dense(1,activation='sigmoid',W_regularizer=l2(0.01)))
#model.add(Dense(2,activation='sigmoid',W_regularizer=l2(0.01),init='normal'))
model.add(Dense(2,init='normal',activation='softmax'))

#编译模型
model.compile(optimizer='adadelta',
             loss='binary_crossentropy',
             metrics=['accuracy'])
#训练模型
history=model.fit(X_train, y_train, batch_size=128, epochs=500, validation_split=0.2,verbose=1)

#获取训练之后的模型权重
w=model.get_weights();
#%%
#画图loss和acc

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('train val loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['loss', 'val_loss'], loc='upper right')


#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('train val acc')
#plt.ylabel('acc')
#plt.xlabel('epoch')
#plt.legend(['acc', 'val_acc'], loc='upper left')
#%%
#预测
y_=model.predict(X_test)
#y_ = y_.clip(min=0.005, max=0.995)  #将结果控制在0.005和0.995之间
#%%
#将预测结果进行输出
import pandas as pd
from keras.preprocessing.image import *

df =pd.DataFrame()
#data生成器
#gen = ImageDataGenerator()
#test_generator = gen.flow_from_directory("test2", (224, 224), shuffle=False, 
#                                         batch_size=16, class_mode=None)
#
#for i, fname in enumerate(test_generator.filenames):
#    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
#    df.set_value(index-1, 'label', y_pred[i])
#输入到文件
for i in range(len(X_test)):
#    print(y_.shape)
    df.set_value(i, 'label0', y_[i][0])
    df.set_value(i,'label1',y_[i][1])
df.head(10)
#df.to_csv(filePath+'pred.csv', index=None)

#print(len(y_))
#print("---------------------")
#print(y_.shape)
#print(sum(y_))
#print(sum(abs(y_)))
#print(y_)

#%%
#模型可视化

#from keras.utils import plot_model
#plot_model(model.summary(), to_file=filePath+'model.png', show_shapes=True)



#from keras.applications.vgg16 import preprocess_input, decode_predictions

#print('Predicted:',decode_psredictions(y_, top=3)[0])
#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#print(correct_prediction)
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(accuracy)




