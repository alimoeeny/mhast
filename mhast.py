import time
import logging

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from PIL import Image
import numpy as np
import wandb
from wandb.keras import WandbCallback

from dataset import DatasetTF

print(f"{tf.__version__}")
project_name = 'foveate'
wandb.login(key="7e48787d23b800f370d524967c31a4fd8c7fb1a1")
wandb.init(project=project_name, entity='alimoeeny')

device = None
if tf.test.is_built_with_cuda:
  devices = tf.config.list_physical_devices('GPU')
  if devices:
    msg = f"We have these devices available, conside using them: {devices}"
    print(msg)
    logging.info(msg)
    tf.config.experimental.set_memory_growth(devices[0], True)
    tf.config.experimental.get_memory_usage('GPU:0')

image_width = 256
batch_size = 13
epochs = 7

input_shape = (image_width, image_width, 3)

inputs = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
x = MaxPooling2D((2, 2), padding="same")(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(inputs, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

train_dataset = DatasetTF(device, root_dir='/Users/ali/Documents/foveate/train', image_width=image_width, max_count=580, shuffle=True)
test_dataset = DatasetTF(device, root_dir='/Users/ali/Documents/foveate/train', image_width=image_width, max_count=580, shuffle=True)

for counter in range(0, 1000):
  test_data = [test_dataset[i] for i in range(counter*batch_size, (counter+1)*batch_size)]
  test_x = tf.convert_to_tensor([test_data[i]['input_img'] for i in range(0, len(test_data))], dtype=tf.float32) / 255.0
  test_y = tf.convert_to_tensor([test_data[i]['output_img'] for i in range(0, len(test_data))], dtype=tf.float32) / 255.0

  data = [train_dataset[i] for i in range(0, counter+batch_size)]
  x = tf.convert_to_tensor([data[i]['input_img'] for i in range(0, len(data))], dtype=tf.float32) / 255.0
  y = tf.convert_to_tensor([data[i]['output_img'] for i in range(0, len(data))], dtype=tf.float32) / 255.0
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

  if counter % 10 != 0:
    continue

  predictions = autoencoder.predict(test_x)

  normalized = np.squeeze(predictions[0,:,:,:])
  normalized /= np.max(np.abs(normalized),axis=0)
  normalized *= (255.0/normalized.max())
  img = Image.fromarray(np.uint8(normalized))
  img.save(f"{time.time()}-{labels[0]}.jpg")