
import numpy as np
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

import outher_Value_and_func as dg
import os
np.random.randint(0 ,25)
os.environ['SM_FRAMEWORK'] = 'tf.keras'
from keras.optimizers import *
import cv2
from datetime import datetime
from os import listdir
from os.path import isfile, join
image_dim = (192, 192)
from tensorflow import keras

def ReadData():
    # ToDo:
    # ReadData
    return images_img,images_mask
def converNumpy(list):
    try:
        array = np.zeros((len(list), list[0].shape[0], list[0].shape[1], list[0].shape[2]), dtype=np.float32)
        for i in range(len(list)):
            array[i, :, :, :] = list[i]
    except:
        array = np.zeros((len(list), list[0].shape[0], list[0].shape[1], 1), dtype=np.float32)
        for i in range(len(list)):
            array[i, :, :, 0] = list[i]
    return array


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score

from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
def result_treshold_05_jaccard_score_f1score(sonuc, y_test, pred_test):
    try:
        y_pred = pred_test[:, :, :].ravel()
        y_true = ( y_test[:, :, :].ravel() >= 0.5) * 1
        y_pred_binary = (y_pred >= 0.5) * 1
        # jaccard_score(y_true, y_pred_binary) * 10000 // 1

        AP=average_precision_score(y_true, y_pred)* 10000 // 1
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        # for idx in range(70):
        #     img=y_test[idx, :, :, 1]
        #     if img.sum()>10:
        #         print(img.sum())
        #         dg.visualize(y_test[idx, :, :, 1], pred_test[idx, :, :, 1])
        # dg.visualize(y_test[4, :, :, 1], pred_test[1, :, :, 1])

        f1_s = f1_score(y_true, y_pred_binary) * 10000 // 1
        iou = jaccard_score(y_true, y_pred_binary) * 10000 // 1
    except:
        iou=0
        f1_s=0
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    # tpr_sensiti=tp/(tp+fp)*10000//1
    # tnr=tn/(tn+fn)*10000//1

    sonuc.append('my_iou_score')
    sonuc.append(str(iou))
    sonuc.append('my_f1-score')
    sonuc.append(str(f1_s))
    sonuc.append('my_roc_auc')
    sonuc.append(str(roc_auc))
    sonuc.append('AP_'+str(AP))

    return sonuc

def GetModel(input_shape, classes, activation):

    import Inc_ZOEA as rn
    model = rn.Inc_ZOEA_MODEL(input_shape=(input_shape),
                                        classes=classes, activation=activation, type='DSEB_5x5_mout')

    modelName = 'Inc_ZOEA'



    return model, modelName


batch_size = 8
Dataset = 'DAGM_class_your_123456'
names = []
for i in [1, 2, 3, 4, 5, 6]:
    names.append('Class' + str(i))
# names = []
# for i in [1]:
#     names.append('Class' + str(i))
images_img, images_mask= ReadData()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images_img,
                                                    images_mask,
                                                    test_size=0.25,
                                                    random_state=20)



print('+++++++++++++++++++++++++++++++++++++++++++++++++')
print('X_train   Shape=', X_train.shape)
print('y_train   Shape=', y_train.shape)

print('X_test   Shape=', X_test.shape)
print('y_test   Shape=', y_test.shape)
print('+++++++++++++++++++++++++++++++++++++++++++++++++')
# ]

X_train = X_train.astype(np.float32) / 255
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32) / 255
y_test = y_test.astype(np.float32)
import segmentation_models as sm

# model_idx_listeeee

pred_test_list = []
pred_test_id = []
# continue
model_idx=15
keras.backend.clear_session()

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if y_train.shape[-1]==1:
    classes = 1
    activation='sigmoid'
else:
    classes = 2
    activation='softmax'

img_row, img_colum, img_channels = X_train.shape[1], X_train.shape[2], X_train.shape[3]
input_shape = (img_row, img_colum, img_channels)

model = None
# tf.tpu.experimental.initialize_tpu_system(tpu)
keras.backend.clear_session()
tf.keras.backend.clear_session()


model, modelName = GetModel(input_shape, classes, activation)
model.summary()

lr = 0.0001

print("---------> lr:", lr)
model.compile(optimizer=Adam(lr=lr), loss=sm.losses.binary_crossentropy,
              metrics=[sm.metrics.iou_score, sm.metrics.f1_score])


print('-----------------------------------------------------------------------------------------------------------------------')
callback, logdir=dg.getColback('Dnm_' + modelName +'_' + str(model_idx) + '_' + Dataset, "./", model)

print('tensorboard --logdir=\''+logdir+'\'')

now = datetime.now()
current_time_bas = now.strftime("%d %m %H:%M:%S.%f")

autoencoder_train = model.fit(X_train, y_train,batch_size=batch_size
                                ,epochs=100,verbose=2)

now = datetime.now()
current_time_son = now.strftime("%H:%M:%S.%f")

now = datetime.now()
current_time_bas_evaluate= now.strftime("%H:%M:%S.%f")
TestSonuc = model.evaluate(X_test, y_test,
                           batch_size=batch_size, verbose=2)

now = datetime.now()
current_time_son_evaluate = now.strftime("%H:%M:%S.%f")
# sonuc=[]
sonuc=[modelName+'_'+'/'+str(model.optimizer.lr.numpy())]

sonuc.append(Dataset)
pred_test= model.predict(X_test, batch_size=batch_size)
sonuc = result_treshold_05_jaccard_score_f1score(sonuc, y_test, pred_test)

print(sonuc)
print(sonuc)





