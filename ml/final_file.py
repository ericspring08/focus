#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, InputLayer, BatchNormalization, Dense, Dropout
from tensorflow import lite
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy


# In[7]:


def build_model(weights_path):
    model_resnet = tf.keras.applications.MobileNetV2(
        include_top=False, weights='imagenet', input_tensor=None,
        input_shape=(224, 224, 3), pooling=None,
    )
    flat = tf.keras.layers.GlobalAveragePooling2D()(
        model_resnet.layers[-1].output)
    d1 = Dense(units=512, activation='relu')(flat)
    d1 = Dense(units=126, activation='relu')(d1)
    d2 = Dense(units=3, activation='softmax')(d1)
    from tensorflow.keras import Model
    model = Model(inputs=model_resnet.inputs, outputs=d2)
    # model.summary()
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
#     model.summary()
    model.load_weights(weights_path)
    return model


# In[8]:


# In[9]:


class get_model():
    def __init__(self, weights_path):
        self.model = build_model(weights_path)

    def predict(self, frame):
        outputs = ["sideway", "attentive", "yawing"]
        frame = cv2.resize(frame, (224, 224))/255.0
        frame = np.expand_dims(frame, axis=0)
        output = self.model.predict(frame)
        index = np.argmax(output)
        attentive_score = output[0][1]
        return outputs[index], attentive_score


# In[10]:

def main():
    final_model = get_model("hackethernet.h5")
    img = cv2.imread("side.jpeg")
    output = final_model.predict(img)
    print(output)


if __name__ == "__main__":
    main()

# In[12]:


# In[13]:


# In[41]:


print(os.getcwd())


# In[ ]:
