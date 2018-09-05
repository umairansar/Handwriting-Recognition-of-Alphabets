# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 15:59:30 2018

@author: sunlight           DUPLICATE but NOW ACCURATE & CORRECT
"""

import matplotlib.pyplot as plt
import numpy as np

# Preparing labels
import os
labels=[]
labels = [f for f in sorted(os.listdir("C:\\Users\\sunlight\\.spyder-py3\\Not MNIST\\notMNIST_small"))]
print(labels)





new_labels = []
for dir in range(0, len(labels)):
    list = os.listdir("C:\\Users\\sunlight\\.spyder-py3\\Not MNIST\\notMNIST_small\\" + labels[dir]) 
    number_files = len(list)
    for i in range(0,number_files):
        new_labels.append(labels[dir])
    print(number_files)
    
print(len(new_labels))

# Importing and preparing images
'''from PIL import Image
import glob
image_list = []
for dir in range(0, len(labels)):
    for filename in glob.glob('C:\\Users\\sunlight\\.spyder-py3\\Not MNIST\\notMNIST_small\\' + labels[dir]): #assuming gif
        im=Image.open(filename)
        image_list.append(im)

print(len(image_list))'''
import cv2
import tensorflow as tf
import glob
images = []
img_dir = "C:\\Users\\sunlight\\.spyder-py3\\Not MNIST\\notMNIST_small\\" # Enter Directory of all images 
for dir in range(0, len(labels)):
    data_path = os.path.join(img_dir + labels[dir],'*g')
    print(data_path)
    files = glob.glob(data_path)
    for f1 in files:
        img = cv2.imread(f1, cv2.IMREAD_GRAYSCALE) # Here (28,28,3) image is converted to (28,28,28)        
        #img = img.reshape(1,784)
        # Start from here                            #using , cv2.IMREAD_GRAY SCALE
        images.append(img)
        
print(len(images))
#images = np.asarray(images)


'''for x in range(0,3):
    dataPath = os.path.join(img_dir + labels[x],'*g')
    image = '''


images = np.asarray(images)
print("+++++++")
print(images.shape)
print("+++++++")

'''
for image in images:
    images[image] = images[image].reshape(?,784)'''
from sklearn.model_selection import train_test_split
image_train, image_test, label_train, label_test = train_test_split(images, new_labels, test_size = 0.2)

#from PIL import Image
#img = Image.open('image.png').convert('LA')


'''
image_train = cv2.cvtColor(image_train, cv2.COLOR_RGB2GRAY)
image_test = cv2.cvtColor(image_test, cv2.COLOR_RGB2GRAY)'''
# Converting simple python list into numpy array
image_train = np.asarray(image_train)
image_test = np.asarray(image_test)
label_train = np.asarray(label_train)# If these 4 lines are not written the under 4 print lines give 
label_test = np.asarray(label_test) # error as simple python list has no shape while numpy has.




print("-----------------")
print(image_train.shape)
print(image_test.shape)
print(label_train.shape)
print(label_test.shape)
print("-----------------")

# Show a random image
random_img = image_test[0]
print(random_img.shape)
plt.imshow(random_img, cmap='gray')
plt.show()

# One-hot_encoding to encode trained_labels
'''from keras.utils import to_categorical 
encoded = to_categorical(label_train)
print(encoded[0])''' # Cant apply as our labels are not integers but char or string.

# One-hot_encoding to encode trained_labels USING LocalBinarizer()
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
transfomed_train_label = encoder.fit_transform(label_train)
transfomed_test_label = encoder.fit_transform(label_test)
print(transfomed_train_label)


# Flatten the images both train and test
flattenedTrainedImages=image_train.reshape(len(image_train),-1) # len(train_images) output: 60000
flattenedTestImages=image_test.reshape(len(image_test),-1) 
print("***********")
print(flattenedTrainedImages.shape) # This is arrau flattened image with 60000 images(rows) and 784 pixels (columns)
print(flattenedTestImages.shape)
print("***********")
print("hello")

# Creating Neural Network
import tensorflow as tf                                                                                   
x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, 8])  # 10 because we have 0 to 9 total possible input number

weights = tf.Variable(tf.random_normal([784,8]))
biases = tf.Variable(tf.random_normal([1,8]))


scores = tf.matmul(x,weights) + biases
pred = tf.nn.softmax(scores)        # to get probabiblity destribution
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = scores))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        sess.run(train, feed_dict={x : flattenedTrainedImages, y : transfomed_train_label})
        print(sess.run(cost,{x:flattenedTrainedImages, y:transfomed_train_label}))

    print(sess.run(accuracy, feed_dict = {x:flattenedTrainedImages, y:transfomed_train_label}))

    randomNumber = np.random.randint(0,2996) 
    image = image_test[randomNumber].reshape(1,784)
    display_image = image.reshape(28,28)
    plt.imshow(display_image, cmap='gray')
    plt.show()
    print("Predicted Alphabet ",encoder.inverse_transform(sess.run(pred, feed_dict={x:image})))




    
    