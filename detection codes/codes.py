from google.colab import drive#connect colab to personal drive
drive.mount('/content/drive/')
pwd# check present working directory
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Lambda,Dense,Flatten,Conv2D
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet  import DenseNet121
from tensorflow.keras.applications.xception import Xception

root_directory='/content/drive/MyDrive/dataset2/Data/train'
if not  os.path.isdir(root_directory+'/NEW_NORMAL'):
   os.makedirs(root_directory+'/NEW_NORMAL')
source='/content/drive/MyDrive/dataset2/Data/train/NORMAL'#where I am copying from
source_name=os.listdir(source)
np.random.shuffle(source_name)
split_ratio=0.43
wanted_files, unwanted_files=np.split(np.array(source_name),[int(len(source_name)*(split_ratio))]) # spliting process 
wanted_files_name=[source+'/'+ name for name in wanted_files]

import shutil # library to copy into new folder
for name in wanted_files_name:
  shutil.copy(name,root_directory+'/NEW_NORMAL')

train_path = '/content/drive/MyDrive/dataset2/Data/train'
test_path = '/content/drive/MyDrive/dataset2/Data/test'

train_normal = pd.DataFrame({"path": os.listdir(train_path + "/NEW_NORMAL"), "label": "NORMAL"})
train_normal["path"] = train_normal["path"].apply(lambda x: train_path + "/NEW_NORMAL/" + x)

train_covid19 = pd.DataFrame({"path": os.listdir(train_path + "/COVID19"), "label": "COVID19"})
train_covid19["path"] = train_covid19["path"].apply(lambda x: train_path + "/COVID19/" + x)

test_normal = pd.DataFrame({"path": os.listdir(test_path + "/NORMAL"), "label": "NORMAL"})
test_normal["path"] = test_normal["path"].apply(lambda x: test_path + "/NORMAL/" + x)

test_covid19 = pd.DataFrame({"path": os.listdir(test_path + "/COVID19"), "label": "COVID19"})
test_covid19["path"] = test_covid19["path"].apply(lambda x: test_path + "/COVID19/" + x)

train_normal

train_covid19

train=pd.concat([train_normal,train_covid19]).reset_index(drop=True) #to combine both normal and covid19 training data sets

test=pd.concat([test_normal,test_covid19]).reset_index(drop=True) #to combine both normal and covid19 testing data sets

train_image_display_normal=[]
for i in range(25):
  normal_number=len(train[train['label']=='NORMAL'])#logical indexing to select normal images 
  image_number=np.random.randint(0,normal_number)
  train_image_display_normal.append(image_number)

#Try the below code
from PIL import Image
plt.figure(figsize=(10, 10))
for i in range(len(train_image_display_normal)):
    ax = plt.subplot(5, 5, i + 1)
    plt.subplots_adjust(top=0.8) 
    img = Image.open(train['path'][train_image_display_normal[i]])
    np_img = np.array(img)
    plt.imshow(np_img)
    plt.axis("off")
    if i==0:
       plt.suptitle('NORMAL IMAGES',y=0.98,
             fontsize = 'xx-large',
             weight = 'extra bold')

train_image_display_covid19=[]
for i in range(25):
  covid19_number=len(train[train['label']=='COVID19'])
  image_number=np.random.randint(0,covid19_number)
  train_image_display_covid19.append(image_number)

from PIL import Image
plt.figure(figsize=(10, 10))
plt.title("COVID19 X-RAY IMAGES")
for i in range(len(train_image_display_covid19)):
    ax = plt.subplot(5, 5, i + 1)
    img = Image.open(train['path'][train_image_display_covid19[i]])
    np_img = np.array(img)
    plt.imshow(np_img)
    plt.axis("off")

# DATA AUGMENTATION

#ImageDataGenerator
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
img=Image.open(train['path'][0])
img=np.array(img)
img=img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
aug_iter=datagen.flow(img, batch_size=1)
fig, axes=plt.subplots(nrows=3, ncols=3, figsize=(15,15))
axes=axes.ravel()
for i in range(9):
  image = next(aug_iter)[0].astype('uint8')
  axes[i].imshow(image)
  axes[i].axis('off')

  #plot image
datagen_train = ImageDataGenerator(rotation_range=10,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   preprocessing_function=preprocess_input,
                                   brightness_range = [0.2, 1.2])

datagen_test = ImageDataGenerator(
    preprocessing_function=preprocess_input)


train_generator_df = datagen_train.flow_from_dataframe(
    dataframe = train,
    x_col = "path",
    y_col = "label",
    class_model = "categorical",
    traget_size = (256, 256),
    batch_size = 32,
    shuffle = True,
    seed = 2020)

test_generator_df = datagen_test.flow_from_dataframe(
    dataframe = test,
    x_col = "path",
    y_col = "label",
    class_model = "categorical",
    traget_size = (256, 256),
    batch_size = 32,
    shuffle = True,
    seed = 2020)

    def resnet50():
  root_model=ResNet50(input_shape=(256,256,3), weights='imagenet', include_top=False)
  for layer in root_model.layers:
    layer.trainable =False
  x=Flatten()(root_model.output)
  prediction = Dense(2, activation ='softmax')(x)
  model=Model(inputs=root_model.input,outputs=prediction)
  return model

  model_resnet50=resnet50()

  model_resnet50.summary()

  model_resnet50.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

  r_resnet50=model_resnet50.fit(train_generator_df, validation_data=test_generator_df, epochs=60, 
                  steps_per_epoch=train.shape[0]//32,
                  validation_steps=test.shape[0]//32, 
                  batch_size=32)

import matplotlib.pyplot as plt 
plt.plot(r_resnet50.history['loss'],label='train loss')#change r to r_resnet50
plt.plot(r_resnet50.history['val_loss'],label='val loss')
plt.legend()
plt.show()
plt.savefig('Lossval_loss')
#plot the accuracy
plt.plot(r_resnet50.history['accuracy'],label='train acc')
plt.plot(r_resnet50.history['val_accuracy'],label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

from tensorflow.keras.models import load_model#change VGG19 to resnet50
model_resnet50.save('model_resnet50.h5')

''' defining a function to make predictions '''
from tensorflow.keras.preprocessing import image
y_pred=[]
def predict(img):
  img=image.load_img(img,target_size=(256,256))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model_resnet50.predict(x)
  pred=np.ndarray.item(np.argmax(preds, axis=1))
  y_pred.append(pred)

  #calling the predict function
for im in test['path']:
  predict(im)

#model evaluation with different metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

test['label']=np.where(test['label']== 'NORMAL', 1,0)
y_test=test['label']

print(f"The accuracy of the model is {accuracy_score(y_test,y_pred)*100:2f}")

print(classification_report(y_test,y_pred))

import seaborn as sns
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, cmap="viridis")