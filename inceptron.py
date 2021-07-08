import time
import os
import gc

import tensorflow as tf
import numpy as np

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

import wandb
from wandb.keras import WandbCallback

from dataset import DatasetTF

save_it_now = False
model_path = '1625683159-autoencoder.save'
load_it_now = False

scale_factor = 8
image_width = int(2048 / scale_factor)
batch_size = 10
epochs = 30

input_shape = (image_width, image_width, 3)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
#predictions = Dense(200, activation='softmax')(x)
predictions = Dense(2)(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.compile(optimizer="adam", loss="mean_squared_error") # loss="binary_crossentropy")
model.summary()

if load_it_now:
  autoencoder = tf.keras.models.load_model(model_path)

train_dataset = DatasetTF(None, root_dir='./train_white_n_center', image_width=image_width, shuffle=True, center_from_name=True)
test_dataset = DatasetTF(None, root_dir='./train_white_n_center', image_width=image_width, shuffle=True, center_from_name=True)


project_name = 'foveate_center_inception3imagenet'
wandb.login(key="7e48787d23b800f370d524967c31a4fd8c7fb1a1")
wandb.init(project=project_name, entity='alimoeeny', name=f"{int(time.time())}")

for counter in range(0, 100000):
  try:
    batch_size = (counter * 2) + 1
    test_data = [test_dataset[i] for i in range(counter*batch_size, (counter+1)*batch_size)]
    test_x = tf.convert_to_tensor([test_data[i]['input_img'] for i in range(0, len(test_data))], dtype=tf.float32) / 255.0
    #test_y = tf.convert_to_tensor([test_data[i]['output_img'] for i in range(0, len(test_data))], dtype=tf.float32) / 255.0
    test_y = tf.convert_to_tensor([test_data[i]['center_coordinate'] for i in range(0, len(test_data))], dtype=tf.float32) / scale_factor

    data = [train_dataset[i] for i in range(0, counter+batch_size)]
    x = tf.convert_to_tensor([data[i]['input_img'] for i in range(0, len(data))], dtype=tf.float32) / 255.0
    #y = tf.convert_to_tensor([data[i]['output_img'] for i in range(0, len(data))], dtype=tf.float32) / 255.0
    y = tf.convert_to_tensor([data[i]['center_coordinate'] for i in range(0, len(data))], dtype=tf.float32) / scale_factor

    labels = [data[n]['identifier'] for n in range(0, len(data))]

    model.fit(
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

    if save_it_now:
      model.save(f'{int(time.time())}-inception3imagenet.save')

    predictions = model.predict(test_x)

    p_coord = np.squeeze(predictions[0,:])
    ty = [int(test_y[0][0]), (test_y[0][1])]
    #file = open(f"./{labels[0]}|{int(ty[0])}|{int(ty[1])}|{int(p_coord[0])}|{int(p_coord[1])}.text", "w")
    file = open(f"./{int(ty[0]-p_coord[0])}|{int(ty[1]-p_coord[1])}|.text", "w")
    file.close()

    del test_data
    del test_x
    del test_y
    del data
    del x
    del y
    del labels
    del predictions
    del ty
    gc.collect()

  except Exception as e:
    # put a breakpoint here
    print(f"Exception happened {e}")
    model.save(f'{int(time.time())}-inception3imagenet.save')
    exit(1)

# at the end ------------------------------------------
model.save(f'{int(time.time())}-inception3imagenet.save')
