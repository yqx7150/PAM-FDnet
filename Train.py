# import library
import tensorflow as tf
import cv2
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
from os import makedirs
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import matplotlib.pyplot as plt

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.config.optimizer.set_jit(True)

########################################################################################################################
# Initialization

seed = 7  # random seed
np.random.seed = seed
tf.random.set_seed(seed)

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1

########################################################################################################################
# Dataset loading

TRAIN_PATH = '../demo/training_dataset/' # 数据集存放位置
data_ids = [filename for filename in os.listdir(TRAIN_PATH) if filename.startswith("x_")]

NUMBER_OF_SAMPLES = int(len(data_ids))
print(NUMBER_OF_SAMPLES)

########################################################################################################################
# Model saving

X_total = np.zeros((NUMBER_OF_SAMPLES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)  # uint8类型存储图像
Y_total = np.zeros((NUMBER_OF_SAMPLES, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
NUMBER_EPOCHS = 200
PATIENCE = 10  # loss does not improve in 10 epochs
MONITOR = 'val_loss'  # loss of validation set

im = TRAIN_PATH.split("d", 1)[1]
FOLDER_NAME = "../demo/ORPAM-withoutNoise" # 模型的存放位置
# if not os.path.exists(FOLDER_NAME):
#     makedirs(FOLDER_NAME)
MODEL_NAME = FOLDER_NAME + 'model.h5'
LOG_NAME = FOLDER_NAME + "logs"

########################################################################################################################
# Image enhancement

print('Resizing training images and masks')
for data, val in enumerate(data_ids):
    ext = val.split("_", 1)[1]  # get the serial number after x_

    xpath = TRAIN_PATH + val
    ypath = TRAIN_PATH + 'y_' + ext

    img = imread(xpath)
    # img = cv2.imread(xpath, cv2.IMREAD_GRAYSCALE)  # [:, :, :IMG_CHANNELS]wowo
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = np.expand_dims(img, axis=2)
    X_total[data] = img  # fill a blank X_train with the value of img

    true_img = imread(ypath)  # [:, :, :IMG_CHANNELS]
    # true_img = cv2.imread(ypath, cv2.IMREAD_GRAYSCALE)
    true_img = resize(true_img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    true_img = np.expand_dims(true_img, axis=2)
    Y_total[data] = true_img

########################################################################################################################
'Divide in training and test data'''

test_split = 0.1  # 10% of the dataset was selected as test set
X_train, X_test, Y_train, Y_test = train_test_split(X_total, Y_total, test_size=test_split, random_state=seed)

Y_pred = np.zeros((len(X_test), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)

X_train, Y_train = shuffle(X_train, Y_train, random_state=seed)

print('Done splitting and shuffling')


########################################################################################################################
# Network functions

def Conv2D_BatchNorm(input, filters, kernel_size, strides, activation, kernel_initializer, padding):
    output = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, activation=activation,
                                    kernel_initializer=kernel_initializer, padding=padding)(input)
    output = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                                beta_initializer='zeros', gamma_initializer='ones',
                                                moving_mean_initializer='zeros',
                                                moving_variance_initializer='ones', beta_regularizer=None,
                                                gamma_regularizer=None,
                                                beta_constraint=None, gamma_constraint=None)(
        output)  # 官方BatchNormalization
    return output


def Conv2D_Transpose_BatchNorm(input, filters, kernel_size, strides, activation, kernel_initializer, padding):
    output = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, activation=activation,
                                             kernel_initializer=kernel_initializer, padding=padding)(input)
    output = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                                beta_initializer='zeros', gamma_initializer='ones',
                                                moving_mean_initializer='zeros',
                                                moving_variance_initializer='ones', beta_regularizer=None,
                                                gamma_regularizer=None,
                                                beta_constraint=None, gamma_constraint=None)(output)
    return output


def DownBlock(input, filters, kernel_size, padding, activation, kernel_initializer):
    output = FD_Block(input, f_in=filters // 2, f_output=filters, k=filters // 8, kernel_size=3, padding='same',
                      # //除法向下取整数
                      activation=activation, kernel_initializer='glorot_normal')
    shortcut = output
    output = DownSample(output, filters, kernel_size, strides=2, padding=padding,
                        activation=activation, kernel_initializer=kernel_initializer)
    return [output, shortcut]


def BrigdeBlock(input, filters, kernel_size, padding, activation, kernel_initializer):
    output = FD_Block(input, f_in=filters // 2, f_output=filters, k=filters // 8, kernel_size=3, padding='same',
                      activation=activation, kernel_initializer='glorot_normal')
    output = UpSample(output, filters, kernel_size, strides=2, padding=padding,
                      activation=activation, kernel_initializer=kernel_initializer)
    return output


def UpBlock(input, filters, kernel_size, padding, activation, kernel_initializer):
    output = Conv2D_BatchNorm(input, filters=filters // 2, kernel_size=1, strides=1, activation=activation,
                              kernel_initializer=kernel_initializer, padding=padding)
    output = FD_Block(output, f_in=filters // 2, f_output=filters, k=filters // 8, kernel_size=3, padding='same',
                      activation=activation, kernel_initializer='glorot_normal')
    output = UpSample(output, filters, kernel_size, strides=2, padding=padding,
                      activation=activation, kernel_initializer=kernel_initializer)
    return output

def FD_Block(input, f_in, f_output, k, kernel_size, padding, activation, kernel_initializer):
    output = input
    for i in range(f_in, f_output, k):
        shortcut = output
        output = Conv2D_BatchNorm(output, filters=f_in, kernel_size=1, strides=1, padding=padding,
                                  activation=activation, kernel_initializer=kernel_initializer)
        output = Conv2D_BatchNorm(output, filters=k, kernel_size=kernel_size, strides=1, padding=padding,
                                  activation=activation, kernel_initializer=kernel_initializer)
        output = tf.keras.layers.Dropout(0.7, seed=seed)(output)
        output = tf.keras.layers.concatenate([output, shortcut])
    return output


def DownSample(input, filters, kernel_size, strides, padding, activation, kernel_initializer):
    output = Conv2D_BatchNorm(input, filters, kernel_size=1, strides=1, activation=activation,
                              kernel_initializer=kernel_initializer, padding=padding)
    output = Conv2D_BatchNorm(output, filters, kernel_size=kernel_size, strides=strides, activation=activation,
                              kernel_initializer=kernel_initializer, padding=padding)
    return output


def UpSample(input, filters, kernel_size, strides, padding, activation, kernel_initializer):
    output = Conv2D_BatchNorm(input, filters, kernel_size=1, strides=1, padding=padding,
                              activation=activation, kernel_initializer=kernel_initializer)
    output = Conv2D_Transpose_BatchNorm(output, filters // 2, kernel_size=kernel_size, strides=strides,
                                        activation=activation,
                                        kernel_initializer=kernel_initializer, padding=padding)
    return output


########################################################################################################################
# Parameters initialization

kernel_initializer = tf.keras.initializers.glorot_normal(seed=seed)
activation = 'relu'  # RELU activation
filters = 16
padding = 'same'
kernel_size = 3
strides = 1

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = inputs


output = Conv2D_BatchNorm(s, filters, kernel_size=kernel_size, strides=strides, activation=activation,
                          kernel_initializer=kernel_initializer, padding=padding)

# Down block (max pooling)
[output, c1] = DownBlock(output, filters * 2 ** 1, kernel_size, padding, activation, kernel_initializer)
[output, c2] = DownBlock(output, filters * 2 ** 2, kernel_size, padding, activation, kernel_initializer)
[output, c3] = DownBlock(output, filters * 2 ** 3, kernel_size, padding, activation, kernel_initializer)
[output, c4] = DownBlock(output, filters * 2 ** 4, kernel_size, padding, activation, kernel_initializer)
[output, c5] = DownBlock(output, filters * 2 ** 5, kernel_size, padding, activation, kernel_initializer)

# Bridge block
output = BrigdeBlock(output, filters * 2 ** 6, kernel_size, padding, activation, kernel_initializer)

# Up block (deconvolution)
output = tf.keras.layers.concatenate([output, c5])
output = UpBlock(output, filters * 2 ** 5, kernel_size, padding, activation, kernel_initializer)
output = tf.keras.layers.concatenate([output, c4])
output = UpBlock(output, filters * 2 ** 4, kernel_size, padding, activation, kernel_initializer)
output = tf.keras.layers.concatenate([output, c3])
output = UpBlock(output, filters * 2 ** 3, kernel_size, padding, activation, kernel_initializer)
output = tf.keras.layers.concatenate([output, c2])
output = UpBlock(output, filters * 2 ** 2, kernel_size, padding, activation, kernel_initializer)
output = tf.keras.layers.concatenate([output, c1])

output = Conv2D_BatchNorm(output, filters, kernel_size=1, strides=1, activation=activation,
                          kernel_initializer=kernel_initializer, padding=padding)
output = FD_Block(output, f_in=filters, f_output=filters * 2, k=filters // 4, kernel_size=3, padding=padding,
                  activation=activation, kernel_initializer=kernel_initializer)

output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding=padding, activation='linear',
                                kernel_initializer=kernel_initializer)(output)
output = tf.keras.layers.Add()([output, s])
output = tf.keras.layers.ReLU()(output)
outputs = output
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

########################################################################################################################
# Optimizer define

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mean_squared_error',
              metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])

########################################################################################################################
# Model checkpoints  # The callback function save the model after every epoch

callbacks = [tf.keras.callbacks.ModelCheckpoint(MODEL_NAME, verbose=1, save_best_only=True),
             tf.keras.callbacks.TensorBoard(log_dir=LOG_NAME),
             tf.keras.callbacks.EarlyStopping(patience=PATIENCE, monitor=MONITOR)]

########################################################################################################################
# Model compiling

results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=8, epochs=NUMBER_EPOCHS, callbacks=callbacks)
print('Model Trained')

########################################################################################################################
# Model evaluation

model.evaluate(X_test, Y_test, verbose=1)
print('Done evaluation')
preds_test = np.zeros((54, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.8)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0] * 0.8):], verbose=1)
preds_test = model.predict(X_test, verbose=1)  # model predictions
print('Done prediction on simulation')
preds_test_int = preds_test.astype(np.uint8)
preds_test_t = tf.convert_to_tensor(preds_test_int)
Y_test_t = tf.convert_to_tensor(Y_test)
X_test_t = tf.convert_to_tensor(X_test)
ssim_test = tf.image.ssim(Y_test_t, preds_test_t, max_val=255)
ssim_test_orig = tf.image.ssim(Y_test_t, X_test_t, max_val=255)
psnr_test = tf.image.psnr(Y_test_t, preds_test_t, max_val=255)
psnr_test_orig = tf.image.psnr(Y_test_t, X_test_t, max_val=255)

print("history: ")
# print(results.history)
# print("loss: ")
print(results.history.get('loss'))
# print("val_loss: ")
print(results.history.get('val_loss'))
plt.plot(results.history.get('loss'), label='loss')
plt.plot(results.history.get('val_loss'), label='val_loss')
plt.legend()

plt.savefig('./figure_loss/0.jpg')
print('SSIM Test')  # the structural similarity of model predictions
print(np.mean(ssim_test.numpy()))

print('PSNR Test')  # the peak signal-to-noise ratio similarity of model predictions
print(np.mean(psnr_test.numpy()))

print('SSIM original')  # the original structural similarity
print(np.mean(ssim_test_orig.numpy()))

print('PSNR original')  # the original peak signal-to-noise ratio similarity
print(np.mean(psnr_test_orig.numpy()))
