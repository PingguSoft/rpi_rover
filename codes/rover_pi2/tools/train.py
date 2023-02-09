import os
import random
import fnmatch
import datetime
import pickle

# data processing
import numpy as np
import pandas as pd

# tensorflow
import tensorflow as tf
import keras
from keras.models import Sequential  # V2 is tensorflow.keras.xxxx, V1 is keras.xxx
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.models import load_model

# sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# imaging
import cv2
from imgaug import augmenters as img_aug
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

_HW = '/device:CPU:0'

###################################################################################################
# image manipulation functions
###################################################################################################
def read_data(data_dir):
    file_list = os.listdir(data_dir)
    image_paths = []
    steering_angles = []
    pattern = "*.png"
    for filename in file_list:
        if fnmatch.fnmatch(filename, pattern):
            image_paths.append(os.path.join(data_dir,filename))
            angle = int(filename[-8:-4])  #00000_0001.png 0001 part of 00000_0001.png is the angle. 0 is go straight
            steering_angles.append(angle)
    return image_paths, steering_angles

def my_imread(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def zoom(image):
    zoom = img_aug.Affine(scale=(1, 1.3))  # zoom from 100% (no zoom) to 130%
    image = zoom.augment_image(image)
    return image

def pan(image):
    # pan left / right / up / down about 10%
    pan = img_aug.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image

def adjust_brightness(image):
    # increase or decrease brightness by 30%
    brightness = img_aug.Multiply((0.7, 1.3))
    image = brightness.augment_image(image)
    return image

def blur(image):
    kernel_size = random.randint(1, 5)  # kernel larger than 5 would make the image way too blurry
    image = cv2.blur(image, (kernel_size, kernel_size))

    return image

def random_flip(image, steering_angle):
    is_flip = random.randint(0, 1)
    if is_flip == 1:
        # randomly flip horizon
        image = cv2.flip(image, 1)
        steering_angle = 0 - steering_angle
    return image, steering_angle

# put it together
def random_augment(image, steering_angle):
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = blur(image)
    if np.random.rand() < 0.5:
        image = adjust_brightness(image)
    image, steering_angle = random_flip(image, steering_angle)
    return image, steering_angle

def img_preprocess(image):
    height, _, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200,66))             # input image size (200,66) Nvidia model
    image = image / 255                             # normalizing, the processed image becomes black for some reason.  do we need this?
    return image

def image_data_generator(image_paths, steering_angles, batch_size, is_training):
    while True:
        batch_images = []
        batch_steering_angles = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            image_path = image_paths[random_index]
            image = my_imread(image_paths[random_index])
            steering_angle = steering_angles[random_index]
            if is_training:
                # training: augment image
                image, steering_angle = random_augment(image, steering_angle)

            image = img_preprocess(image)
            batch_images.append(image)
            batch_steering_angles.append(steering_angle)

        yield (np.asarray(batch_images), np.asarray(batch_steering_angles))


###################################################################################################
# data analysis functions
###################################################################################################
def show_data_hist(image_paths, steering_angles, num_of_bins):
    # show 1 image
    image_index = 20
    plt.imshow(Image.open(image_paths[image_index]))
    print("image_path: %s" % image_paths[image_index] )
    print("steering_Angle: %d" % steering_angles[image_index] )
    df = pd.DataFrame()
    df['ImagePath'] = image_paths
    df['Angle'] = steering_angles
    plt.title("show 20th image")

    # show histogram
    hist, bins = np.histogram(df['Angle'], num_of_bins)
    fig, axes = plt.subplots(1,1, figsize=(12,4))
    axes.hist(df['Angle'], bins=num_of_bins, width=1, color='blue')
    axes.set_title("Angle histogram in data")
    plt.show()

def show_data(image_paths, steering_angles, num_of_bins, X_train, X_valid, y_train, y_valid):
    image_index = 20

    # plot the distributions of train and valid, make sure they are consistent
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    axes[0].hist(y_train, bins=num_of_bins, width=1, color='blue')
    axes[0].set_title('Training Data')
    axes[1].hist(y_valid, bins=num_of_bins, width=1, color='red')
    axes[1].set_title('Validation Data')

    #
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    image_orig = my_imread(image_paths[image_index])
    image_processed = img_preprocess(image_orig)
    axes[0].imshow(image_orig)
    axes[0].set_title("orig")
    axes[1].imshow(image_processed)
    axes[1].set_title("processed")

    # show a few randomly augmented images
    ncol = 2
    nrow = 5
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 50))

    for i in range(nrow):
        rand_index = random.randint(0, len(image_paths) - 1)
        image_path = image_paths[rand_index]
        steering_angle_orig = steering_angles[rand_index]

        image_orig = my_imread(image_path)
        image_aug, steering_angle_aug = random_augment(image_orig, steering_angle_orig)

        axes[i][0].imshow(image_orig)
        axes[i][0].set_title("original, angle=%s" % steering_angle_orig)
        axes[i][1].imshow(image_aug)
        axes[i][1].set_title("augmented, angle=%s" % steering_angle_aug)

    X_train_batch, y_train_batch = next(image_data_generator(X_train, y_train, nrow, True))
    X_valid_batch, y_valid_batch = next(image_data_generator(X_valid, y_valid, nrow, False))
    print("Training data: %d\nValidation data: %d" % (len(X_train_batch), len(X_valid_batch)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 6))
    fig.tight_layout()

    for i in range(nrow):
        axes[i][0].imshow(X_train_batch[i])
        axes[i][0].set_title("training, angle=%s" % y_train_batch[i])
        axes[i][1].imshow(X_valid_batch[i])
        axes[i][1].set_title("validation, angle=%s" % y_valid_batch[i])
    plt.show()


###################################################################################################
# training & analysis
###################################################################################################
def nvidia_model():
    model = Sequential(name='Nvidia_Model')

    # elu=Expenential Linear Unit, similar to leaky Relu
    # skipping 1st hiddel layer (nomralization layer), as we have normalized the data

    # Convolution Layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.2))  # not in original model. added for more robustness
    model.add(Conv2D(64, (3, 3), activation='elu'))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dropout(0.2))  # not in original model. added for more robustness
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))

    # output layer: turn angle (from 45-135, 90 is straight, <90 turn left, >90 turn right)
    model.add(Dense(1))

    # since this is a regression problem not classification problem,
    # we use MSE (Mean Squared Error) as loss function
    optimizer = Adam(learning_rate=1e-3)  # lr is learning rate
    model.compile(loss='mse', optimizer=optimizer)

    return model

def train(model_output_dir, X_train, X_valid, y_train, y_valid):
    # saves the model weights after each epoch if the validation loss decreased
    with tf.device(_HW):
        model = nvidia_model()
        print(model.summary())

        checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_output_dir, 'rover_pi2_check.h5'),
                                                              verbose=1, save_best_only=True)
        history = model.fit(image_data_generator(X_train, y_train, batch_size=100, is_training=True),
                            steps_per_epoch=100,
                            epochs=10,
                            validation_data=image_data_generator(X_valid, y_valid, batch_size=100, is_training=False),
                            validation_steps=200,
                            verbose=1,
                            shuffle=1,
                            callbacks=[checkpoint_callback])
        # always save model output as soon as model finishes training
        model.save(os.path.join(model_output_dir, 'rover_pi2_final.h5'))
        date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        history_path = os.path.join(model_output_dir, 'history.pickle')
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)

def train_analysis(model_output_dir):
    history = None
    history_path = os.path.join(model_output_dir, 'history.pickle')
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    # plt.plot(history['accuracy'], color='blue')
    # plt.plot(history['val_accuracy'], color='red')
    # plt.legend(["training accuracy", "validation accuracy"])
    # plt.title("accuracy analysis")

    plt.plot(history['loss'], color='blue')
    plt.plot(history['val_loss'], color='red')
    plt.legend(["training loss", "validation loss"])
    plt.title("loss analysis")
    plt.show()

def summarize_prediction(Y_true, Y_pred):
    mse = mean_squared_error(Y_true, Y_pred)
    r_squared = r2_score(Y_true, Y_pred)
    print(f'mse       = {mse:.2}')
    print(f'r_squared = {r_squared:.2%}')

def predict_and_summarize(X, Y, model_output_dir):
    with tf.device(_HW):
        model = load_model(f'{model_output_dir}/rover_pi2_check.h5')
        Y_pred = model.predict(X)
        summarize_prediction(Y, Y_pred)
    return Y_pred

def representative_dataset():
    for _ in range(200):
      data = np.random.rand(1, 66, 200, 3)
      yield [data.astype(np.float32)]

def convert_to_tflite(model_file, tflite_file, is_quant):
    with tf.device(_HW):
        model = load_model(model_file)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if is_quant:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                # tf.lite.OpsSet.SELECT_TF_OPS,           # enable TensorFlow ops.
                tf.lite.OpsSet.TFLITE_BUILTINS,         # enable TensorFlow Lite ops.
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        with open(tflite_file, "wb") as f:
            f.write(tflite_model)


###################################################################################################
# main
###################################################################################################
if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind': lambda x: "%.4f" % x})
    pd.set_option('display.width', 300)
    pd.set_option('display.float_format', '{:,.4f}'.format)
    pd.set_option('display.max_colwidth', 200)

    print(f'tf.__version__: {tf.__version__}')
    print(f'keras.__version__: {keras.__version__}')

    # image_paths, steering_angles = read_data('.rec/1224_002556_img/')
    # X_train, X_valid, y_train, y_valid = train_test_split( image_paths, steering_angles, test_size=0.2)
    # print("Training data: %d\nValidation data: %d" % (len(X_train), len(X_valid)))
    #
    # # show_data_hist(image_paths, steering_angles, 25)
    # # show_data(image_paths, steering_angles, 25, X_train, X_valid, y_train, y_valid)
    #
    model_output_dir = '.train_outputs'
    # # train(model_output_dir, X_train, X_valid, y_train, y_valid)
    #
    # n_tests = 100
    # X_test, y_test = next(image_data_generator(X_valid, y_valid, n_tests, False))
    # y_pred = predict_and_summarize(X_test, y_test)
    # train_analysis(model_output_dir)

    #convert_to_tflite(f'{model_output_dir}/rover_pi2_final.h5', f'{model_output_dir}/rover_pi2_final.tflite', False)
    #convert_to_tflite(f'{model_output_dir}/rover_pi2_final.h5', f'{model_output_dir}/rover_pi2_final_quant8.tflite', True)
    convert_to_tflite(f'{model_output_dir}/rover_pi2_final.h5', f'{model_output_dir}/rover_pi2_final_quant8.tflite',
                      True)
