import tensorflow as tf
import glob
import datetime
import numpy as np
import base64

from tensorflow.python.saved_model import tag_constants

labelGlob = glob.glob("dataset3/train/*")
labelMaster = np.ndarray(shape=(9), dtype=np.dtype('S20'))

loop = 0
for folder in labelGlob:
    labelKey = folder.split("/")[2]
    labelMaster.itemset(loop, labelKey)
    loop = loop + 1

predictImages = np.ndarray(shape=(1,40000))
predictLabels = np.ndarray(shape=(1,9))

sess = tf.InteractiveSession()
tf.logging.set_verbosity(tf.logging.INFO)

init = tf.global_variables_initializer()
sess.run(init)

# Load the image we want to do a prediction for:
# Build the pixel map
file_contents = tf.read_file("dataset3/test/dccomics/image_02266.jpg")
readImage = tf.image.decode_jpeg(file_contents, channels=1)

data_uri = open("dataset3/test/dccomics/image_02266.jpg", "rb").read()
file_results = base64.standard_b64encode(data_uri)

print ("file_results: ", file_results)

image = tf.image.resize_images(readImage, [200, 200])

with sess.as_default():
    flatImage = image.eval().ravel()
flatImage = np.multiply(flatImage, 1.0 / 255.0)
predictImages[0] = flatImage
predictLabels[0] = [0, 0, 0, 0, 0, 0, 0, 0, 0]

flatImage2 = tf.reshape(image, [-1])
flatImage3 = sess.run(flatImage2)
flatImage3 = np.multiply(flatImage3, 1.0 / 255.0)

print ("flatImage:  ", flatImage)
print ("flatImage2: ", flatImage2)
print ("flatImage3: ", flatImage3)

print("Start Time: ", datetime.datetime.now())
print ("Loading the graph? ...")

import_dir = 'data/model/1'
tf.saved_model.loader.load(sess, [tag_constants.SERVING], import_dir)

print ("Load complete?")
print("End Time:   ", datetime.datetime.now())

x = sess.graph.get_tensor_by_name("x:0")
y_ = sess.graph.get_tensor_by_name("y_:0")
keep_prob = tf.placeholder(tf.float32)

keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
predictator = sess.graph.get_tensor_by_name("predictator:0")

predict_it = sess.run(predictator, feed_dict={x: predictImages, y_: predictLabels, keep_prob:1.0})

prediction = labelMaster[predict_it]

print ("prediction: ", bytes.decode(prediction[0]))

# for tensor in tf.get_default_graph().as_graph_def().node:
#     print ("tensor.name: ", tensor.name)

