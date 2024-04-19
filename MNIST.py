#!/usr/bin/env python
# coding: utf-8

# ### Name: Hari Vengadesh
# 
# #### Homework 3

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# In[2]:


import numpy as np
import pandas as pd
from datetime import datetime

import PIL
import PIL.Image
import pathlib

from matplotlib import pyplot as plt
import matplotlib as mpl
import tensorflow as tf

from sklearn.metrics import confusion_matrix
import seaborn as sns

print(tf.__version__)


# In[3]:


print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda)
print(tf.test.gpu_device_name())
print(tf.config.get_visible_devices())


# #### Parameters

# In[4]:


AUTOTUNE = tf.data.AUTOTUNE


# In[5]:


batch_size = 32
img_height = 28
img_width = 28


# #### Data

# In[6]:


data_dirpathname = r'E:\Data Science\ECE565 - Machine Learning\datasets\mnist\trainingSet\trainingSet'
data_dir = pathlib.Path(data_dirpathname)

class_names = os.listdir(data_dir)
num_classes = len(class_names)


# ## 1. 2.a) Code that will count number of classes
# ## 1. 2.b) Number of images in each class

# In[7]:


images_per_class = {}
for class_name in class_names:
    images_per_class[class_name] = len(os.listdir(os.path.join(data_dir, class_name)))

print("Number of classes:", num_classes)
print("Number of images in each class:", images_per_class)


# In[8]:


image_count =  len(list(data_dir.glob('*/*.jpg')))
print('Image count:', image_count)


# In[9]:


one = list(data_dir.glob('1/*'))
PIL.Image.open(str(one[0]))


# In[10]:


sample_image = PIL.Image.open(str(one[1]))
img = np.asarray(sample_image)
img.shape


# #### Setup Dataset Pipeline

# In[11]:


list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
list_ds


# In[12]:


for f in list_ds.take(5):
    print(f.numpy())


# ## 1. 1. Data must be split in train/test and validation set

# In[13]:


test_size = int(image_count*0.2)
train_ds = list_ds.skip(test_size)
val_ds = list_ds.take(test_size)


# In[14]:


print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())


# #### Helper Functions

# In[15]:


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)


# In[16]:


def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [img_height, img_width])


# In[17]:


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# In[18]:


# Set 'num_parallel_calls' so multiple images are loaded/processed in parallel.

train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)


# ## 1. 2.c) Image resized to 101x101xNo_Of_channels.
# 
# ## 1. 2.d) Automatic data label extraction based on sub-directory name.
# 
# ## 1. 2.e) Display one batch of image/label using the dataset api.

# In[19]:


for image, label in train_ds.take(1):
    print("Image shape:", image.numpy().shape)
    print("Label:", class_names[label.numpy()])


# In[20]:


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size=64)
    return ds


# In[21]:


train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)


# #### Test Dataset Pipeline

# In[22]:


image_batch, label_batch = next(iter(train_ds))


# In[23]:


plt.figure(figsize=(10,10))
for i in range(9):
    ax = plt.subplot(3, 3, i+1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    label = label_batch[i]
    plt.title(class_names[label])


# ## 2. Model

# In[24]:


model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(img_height, img_width, 3), name='Input_Shape_Layer'),
    tf.keras.layers.Rescaling(1./255),  # Normalize pixel values
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])


# In[25]:


tf.keras.utils.plot_model(model=model, rankdir="LR", dpi=72, show_shapes=True)


# In[26]:


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# ### Training / Validation Cycle

# In[27]:


logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# ## 3. Train/Validation
# 
# ## 3. 1. Early stopping

# In[28]:


early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=5)


# In[29]:


model.fit(
    train_ds,
    validation_data=val_ds, 
    epochs=101,
    callbacks=[tensorboard_callback, early_stopping_callback],
    verbose=1
)


# #### Evaluate Trained Model

# ## 4. Model Evaluation
# 
# ## 4. 2. Accuracy Metrics

# In[30]:


test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print('\nTest Accuracy:', test_acc)


# In[31]:


all_predictions = []
all_labels = []


# ## 4. 1. Confusion Matrix

# In[32]:


for images, labels in val_ds:
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = labels.numpy()

    all_predictions.extend(predicted_classes)
    all_labels.extend(true_classes)


# In[33]:


cm = confusion_matrix(all_labels, all_predictions)
class_names_array = np.array(class_names)


# In[34]:


plt.figure(figsize=(10, 8))  # Increase figure size to make room for labels
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=class_names, yticklabels=class_names)

# Set tick labels appearance
plt.xticks(rotation=45, ha='right', fontsize=10)  # Adjust font size or rotation as needed
plt.yticks(fontsize=10)

# Colorbar label and title
plt.title('Confusion Matrix', fontsize=14)  # Increase title font size
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

# Loop over data dimensions and create text annotations with contrasting colors
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j+0.5, i+0.5, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')

plt.show()


# #### Make Predictions

# In[35]:


probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])


# In[36]:


predictions = probability_model.predict(val_ds)


# In[37]:


predictions[0]


# In[38]:


class_names[np.argmax(predictions[0])]


# In[39]:


image_batch, label_batch = next(iter(val_ds))


# In[40]:


label_batch


# In[41]:


predictions = probability_model.predict(image_batch)


# In[42]:


np.argmax(predictions, axis=1)


# In[43]:


image_batch[0].shape


# In[44]:


predictions_prob = probability_model.predict(image_batch)
predictions = np.argmax(predictions_prob, axis=1)


# ## 5. Model Predictions

# In[45]:


plt.figure(figsize=(20, 20))
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['axes.titlepad'] = 5

for i in range(16):
    ax = plt.subplot(4, 4, i+1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    true_label = class_names[label_batch[i]]
    predicted_label = class_names[predictions[i]]
    title_text = f'actual={true_label},  predicted={predicted_label}'
    plt.title(title_text, wrap=True, backgroundcolor='white', fontsize=18) 
    plt.axis('off')

plt.tight_layout(pad=1.0)
plt.show()


# ## 3. 2. Tensorboard

# In[46]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir {logdir}')


# In[ ]:




