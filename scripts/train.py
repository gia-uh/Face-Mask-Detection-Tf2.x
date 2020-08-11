# %%
import tensorflow as tf
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# %%
batch_size = 32
epoch = 2
num_classes = 2
step_per_epoch = len(glob('../data/train/*/*'))//batch_size

image_h, image_w = 128, 128
classes = sorted(['with_mask','without_mask'])

# %% 
# # **Reading Images**
# - We need to apply transformations on the input image before defining inputs for the model architecture.
# - Transformations are done so as to ensure that the inputs to the network are of the right size and respective values sit in a similar range.
# - Input image was resized to (128,128) for (height, width) and values in the image tensor were scaled to the range of (-1,1).
# - Entire dataset was split into training and test set for cross validation. Split ratio was chose such that 90 percent of the dataset is part of the training set and 10 percent for test set.

def read_img(image_path):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_image(img, channels=3)
  img.set_shape([None,None,3])
  img = tf.image.resize(img, [image_w, image_h])
  img = img/127.5-1
  return img

label_map = {v:i for i, v in enumerate(classes)}

images = glob('../data/train/*/*')
np.random.shuffle(images)

labels = [label_map[x.split('/')[-2]] for x in images]

(train_images, test_images, train_labels, test_labels) = train_test_split(images, labels,
	test_size=0.10, stratify=labels, random_state=42)

#reading image and label
def load_data(image_path, label):
  image = read_img(image_path)
  return image, label

# %% 
# # **Data Generator**
# 
# - Using a data generator, we can batch individual samples into a unified set.
# - Data generator spawns processes for loading data to input while training the model.
# - This function also provides parallel loading for batches in the training data using shared memory and separate process.
# 

def data_generator(features,labels):
  dataset = tf.data.Dataset.from_tensor_slices((features,labels))
  dataset = dataset.shuffle(buffer_size=100)
  autotune = tf.data.experimental.AUTOTUNE
  dataset = dataset.map(load_data, num_parallel_calls=autotune)
  dataset = dataset.batch(batch_size=batch_size)
  dataset = dataset.repeat()
  dataset = dataset.prefetch(autotune)
  return dataset

# %%
def show_img(dataset):
  plt.figure(figsize=(15,15))
  for i in range(8):
    for val in dataset.take(1):
      img  = (val[0][i]+1)*127.5
      plt.subplot(4,2,i+1)
      plt.imshow(tf.cast(img,tf.uint8))
      plt.title(val[1][i].numpy())
      plt.subplots_adjust(hspace=1)
  plt.show()

train_dataset = data_generator(train_images, train_labels)
test_dataset = data_generator(test_images, test_labels)

# (!) Run this to show some example images
# show_img(train_dataset)

# %%
# ## Downloading the Xception model
 
base_model = tf.keras.applications.Xception(include_top=False,
                                       input_shape=(None, None, 3),
                                       weights='imagenet')
base_model.trainable = False
layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
layer = tf.keras.layers.Dense(1024, activation='relu')(layer)
layer = tf.keras.layers.Dropout(0.5)(layer)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(layer)
model = tf.keras.models.Model(base_model.inputs,output)
model.summary()

# %% 
# # **Model fitting**
# 
# Implementation details for classification architecture:
# - ***Batch Size***: 32 
# - ***Epochs***: 2 
# - ***Learning rate***: 1e-4 (with decay of 1e-4 / epoch)
# - ***Gradient Descent Optimizer***: Adam
# - ***Loss function***: Sparse categorical cross entropy
# - ***Criterion for evaluation*** (metric): F1-score

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4, decay=1e-4 / epoch),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('../data/mask_classification_model.tmp.h5', 
                                       save_best_only=True, 
                                       save_weights_only=True,
                                       monitor='val_loss')]

model.fit(train_dataset,
          # batch_size=batch_size,
          epochs=epoch,steps_per_epoch=step_per_epoch,
          validation_data=test_dataset,
          validation_steps=1,
          callbacks=callbacks)

# %%
model.save('mask_classification_model.h5')
