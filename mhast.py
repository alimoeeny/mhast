import time
import logging

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Reshape, Concatenate, Dropout
from PIL import Image
import numpy as np
from tensorflow.python.keras.layers.core import Flatten
import wandb
from wandb.keras import WandbCallback

from dataset import DatasetTF

print(f"{tf.__version__}")

device = None
if tf.test.is_built_with_cuda:
  devices = tf.config.list_physical_devices('GPU')
  if devices:
    msg = f"We have these devices available, conside using them: {devices}"
    print(msg)
    logging.info(msg)
    tf.config.experimental.set_memory_growth(devices[0], True)
    tf.config.experimental.get_memory_usage('GPU:0')

scale_factor = 8
image_width = int(2048 / scale_factor)
batch_size = 1
epochs = 30

input_shape = (image_width, image_width, 3)

inputs = Input(shape=input_shape)
x = Conv2D(3, (3, 3), activation="relu", padding="same")(inputs)
x = Dropout(.2)(x)
#x = MaxPooling2D((2, 2), padding="same")(x)
#x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
#x = MaxPooling2D((2, 2), padding="same")(x)

# x = Concatenate(axis=3)([inputs, x])
x = Flatten()(x)
x = Dense(64*64)(x)
x = Dense(2)(x)

# Autoencoder
autoencoder = Model(inputs, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

train_dataset = DatasetTF(device, root_dir='/Users/ali/Documents/foveate/train_white_n_center', image_width=image_width, max_count=580, shuffle=True, center_from_name=True)
test_dataset = DatasetTF(device, root_dir='/Users/ali/Documents/foveate/train_white_n_center', image_width=image_width, max_count=580, shuffle=True, center_from_name=True)


project_name = 'foveate'
wandb.login(key="7e48787d23b800f370d524967c31a4fd8c7fb1a1")
wandb.init(project=project_name, entity='alimoeeny')

for counter in range(0, 1000):
  batch_size = counter + 1
  test_data = [test_dataset[i] for i in range(counter*batch_size, (counter+1)*batch_size)]
  test_x = tf.convert_to_tensor([test_data[i]['input_img'] for i in range(0, len(test_data))], dtype=tf.float32) / 255.0
  #test_y = tf.convert_to_tensor([test_data[i]['output_img'] for i in range(0, len(test_data))], dtype=tf.float32) / 255.0
  test_y = tf.convert_to_tensor([test_data[i]['center_coordinate'] for i in range(0, len(test_data))], dtype=tf.float32) / scale_factor

  data = [train_dataset[i] for i in range(0, counter+batch_size)]
  x = tf.convert_to_tensor([data[i]['input_img'] for i in range(0, len(data))], dtype=tf.float32) / 255.0
  #y = tf.convert_to_tensor([data[i]['output_img'] for i in range(0, len(data))], dtype=tf.float32) / 255.0
  y = tf.convert_to_tensor([data[i]['center_coordinate'] for i in range(0, len(data))], dtype=tf.float32) / scale_factor

  labels = [data[n]['identifier'] for n in range(0, len(data))]

  autoencoder.fit(
    x=x,
    y=y,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(test_x, test_y),
    callbacks=[
      WandbCallback(labels=[labels], verbose=0, save_model=False, ),
    ]
  )

  # if counter % 5 != 0:
  #   continue

  predictions = autoencoder.predict(test_x)

  p_coord = np.squeeze(predictions[0,:])
  ty = [int(test_y[0][0]), (test_y[0][1])]
  file = open(f"./{labels[0]}|{ty[0]}|{ty[1]}|{int(p_coord[0])}|{int(p_coord[1])}.text", "w")
  file.close()

  # normalized = np.squeeze(predictions[0,:,:,:])
  # mx = np.max(normalized,axis=0)
  # mn = np.mom(normalized,axis=0)
  # normalized = (normalized - mn)
  # normalized = (normalized / max) * 255
  # img = Image.fromarray(np.uint8(normalized))
  # img.save(f"{time.time()}-{labels[0]}.jpg")