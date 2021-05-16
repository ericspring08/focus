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


def eyes(img, faceCascade, eyeCascade):
    print(img.shape)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        # flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    # print("Found {0} faces!".format(len(faces)))
    if len(faces) > 0:
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frame_tmp = img[faces[0][1]:faces[0][1] + faces[0]
                        [3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
        frame = frame[faces[0][1]:faces[0][1] + faces[0]
                      [3], faces[0][0]:faces[0][0] + faces[0][2]:1]
        eyes = eyeCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        if len(eyes) == 0:
            return "sleepy"
    return "not_sleepy"


# In[8]:


# In[9]:

class get_model():
    def __init__(self, weights_path):
        self.model = build_model(weights_path)
        self.faceCascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_alt.xml')
        self.eyeCascade = cv2.CascadeClassifier(
            'haarcascade_eye_tree_eyeglasses.xml')

    def predict(self, frame):
        sleepy_behaviour = eyes(frame, self.faceCascade, self.eyeCascade)
        outputs = ["sideway", "attentive", "yawing"]
        frame = cv2.resize(frame, (224, 224))/255.0
        frame = np.expand_dims(frame, axis=0)
        output = self.model.predict(frame)
        index = np.argmax(output)
        attentive_score = output[0][1]

        return outputs[index], attentive_score, sleepy_behaviour

# In[10]:


def main():
    final_model = get_model("hackethernet.h5")
    img = cv2.imread("front.jpeg")
    output = final_model.predict(img)
    print(output)


if __name__ == "__main__":
    main()

# In[12]:


# In[13]:


# In[41]:


print(os.getcwd())


# In[ ]:
