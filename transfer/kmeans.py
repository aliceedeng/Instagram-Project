import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import urllib
import instaloader
from instaloader import *
from skimage import io
import os
from PIL import *
import requests
from io import BytesIO
import PIL.ImageStat
import pickle 
import io as urlIo
import ssl

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

#def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    #resp = urllib.urlopen(url)
    #image = np.asarray(bytearray(resp.read()), dtype="uint8")
    #image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
   # return image

#L = instaloader.Instaloader()

#newp = os.listdir('/scratch/jamaalhay/PostsNew')

def getclust(pot):
    clt = KMeans(n_clusters=3)
    dictt = {}

# This restores the same behavior as before.
    context = ssl._create_unverified_context()
    post = load_structure_from_file(L.context, "/scratch/jamaalhay/PostsNew/" + pot + "")
    #img = io.imread(post.url, context = context)
    filee = urlIo.BytesIO(urllib.request.urlopen(post.url, context = context).read())
    img = Image.open(filee)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = img.reshape((img.shape[0] * img.shape[1],3))
    clt.fit(img)
    hist = find_histogram(clt)
    dictt['col1'] = [hist[0], clt.cluster_centers_[0]]
    dictt['col2'] = [hist[1], clt.cluster_centers_[1]]
    dictt['col3'] = [hist[2], clt.cluster_centers_[2]]
    return dictt
fav = pickle.load(open("save_objectDetect.p", "rb" ))
L = instaloader.Instaloader()
newp = os.listdir('/scratch/jamaalhay/PostsNew')
dict_tot = {}
count = 0
for pot in fav:
    pott = str(pot) + '.json'
    if count % 50 == 0:
        print(count)
    #try:
    dictor = getclust(pott)
    #except:
    #    continue
    dict_tot[pot] = dictor
    count += 1
pickle.dump(dict_tot, open( "save_colours.p", "wb" ) )
