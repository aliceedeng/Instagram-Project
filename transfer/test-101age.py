#!/usr/bin/env python
# coding: utf-8

# In[1]:


import CNN2Head_input2 as CNN2Head_input
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import BKNetStyle2 as BKNetStyle
from const import *
import cv2

#from IPython.display import Image, display
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt


# In[2]:
from skimage import io
import instaloader
from instaloader import *
import urllib.request
import pickle
import ssl
import io as urlIo
from io import BytesIO
from PIL import *
import multiprocessing
from multiprocessing import Pool
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# In[4]:


def predict(pot, dictt):
    sess = tf.InteractiveSession()
    L = instaloader.Instaloader()
    #for post in posts:
    PATH1 = '/scratch/jamaalhay/PostsNew/' + str(pot)
    post = load_structure_from_file(L.context, PATH1)
    #img = io.imread(post.url)
    context = ssl._create_unverified_context()
    filee = urlIo.BytesIO(urllib.request.urlopen(post.url, context = context).read())
    img = Image.open(filee)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #print(post.url)
    #urllib.request.urlretrieve(post.url, "/scratch/jamaalhay/PostsNew/00000001.jpg")
    #PATH = PATH + "00000001.jpg"
    #display(Image(filename = PATH, width=100, height=100))
    count_s = 0
    count_g = 0
    result = detector.detect_faces(img)
    try:
        face_position = result[0].get('box')
    except:
        return '1'
    try:
        x_coordinate = face_position[0]
        y_coordinate = face_position[1]
        w_coordinate = face_position[2]
        h_coordinate = face_position[3]
        img = img[y_coordinate:y_coordinate+h_coordinate, x_coordinate:x_coordinate+w_coordinate]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(48, 48))
        img = (img - 128) / 255.0
        T = np.zeros([48, 48, 1])
        T[:, :, 0] = img
        test_img = []
        test_img.append(T)
        test_img = np.asarray(test_img)
    except:
        return '1'
    for i in range(25):
        init = tf.global_variables_initializer()
        sess.run(init)
        predict_y_smile_conv = sess.run(y_smile_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})
        #predict_y_gender_conv = sess.run(y_gender_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})
        #predict_y_age_conv = sess.run(y_age_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})
        smile_label = "No" if np.argmax(predict_y_smile_conv)==0 else "Yes"
        #gender_label = "Female" if np.argmax(predict_y_gender_conv)==0 else "Male"
        if smile_label == "Yes":
            count_s += 1
        if count_s >= 13:
            smile_label = "Yes"
        else:
            smile_label = "No"
    dictt[pot[:-5]] = smile_label


# In[5]:
detector = MTCNN()
x, y_, mask = BKNetStyle.Input()
y_smile_conv,  phase_train, keep_prob = BKNetStyle.BKNetModel(x)

if __name__ == '__main__':
    jobs = []
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    favorite = pickle.load( open( "save_objectDetect.p", "rb" ) )
   # x, y_, mask = BKNetStyle.Input()
    #y_smile_conv,  phase_train, keep_prob = BKNetStyle.BKNetModel(x)
    #smile_loss, l2_loss, loss = BKNetStyle.selective_loss(y_smile_conv, y_, mask)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    posts = os.listdir('/scratch/jamaalhay/PostsNew')
    sess = tf.InteractiveSession(config = config)
    print('Restore model')
    #with tf.Session() as sess1:
    saver = tf.train.Saver()
    saver = tf.train.import_meta_graph('./save/current6/model-age101.ckpt.meta')
    saver.restore(sess, './save/current6/model-age101.ckpt')
    sess.close()
    count = 0
    multiprocessing.set_start_method('spawn')
    manager = multiprocessing.Manager()
    dictt = manager.dict()
    for post in favorite:
        pot = str(post) + '.json'
        count += 1
        if count % 20  == 1:
            print(count)
        #try:
        p = multiprocessing.Process(target=predict, args = (pot,dictt))
        p.start()
        p.join()
    pickle.dump(dict(dictt), open( "save_smile2.p", "wb" ) )


#if __name__ == '__main__':
    #L = instaloader.Instaloader()
    #x, y_, mask = BKNetStyle.Input()

    #y_smile_conv,  phase_train, keep_prob = BKNetStyle.BKNetModel(x)
    #smile_loss, l2_loss, loss = BKNetStyle.selective_loss(y_smile_conv, y_, mask)

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #detector = MTCNN()
    #posts = os.listdir('/scratch/jamaalhay/PostsNew')

    #multiprocessing.set_start_method('spawn')
    #multiprocessing.Pool().map(runner, range(1))
    #p = multiprocessing.Process(target=run_tensorflow)
    #p.start()
    #p.join()



