import tensorflow as tf
import glob
import datetime
import numpy as np

from tensorflow.python.saved_model import tag_constants

class ImagePredictor:
    def __init__(self):
        # ************************************************************************************************
        print("Start Time: ", datetime.datetime.now())
        print("Loading the graph? ...")

        self.labelGlob = glob.glob("dataset3/train/*")
        self.labelMaster = np.ndarray(shape=(9), dtype=np.dtype('S20'))

        loop = 0
        for folder in self.labelGlob:
            labelKey = folder.split("/")[2]
            self.labelMaster.itemset(loop, labelKey)
            loop = loop + 1

        self.predictImages = np.ndarray(shape=(1, 40000))
        self.predictLabels = np.ndarray(shape=(1, 9))

        self.import_dir = 'data/model/1'

        print("Initialization Complete")
        # ************************************************************************************************

    def predict_image(self, image_name):
        # ************************************************************************************************
        print("Start Time: ", datetime.datetime.now())
        print("Loading the graph? ...")

        sess = tf.Session()
        tf.logging.set_verbosity(tf.logging.INFO)

        init = tf.global_variables_initializer()
        sess.run(init)

        tf.saved_model.loader.load(sess, [tag_constants.SERVING], self.import_dir)

        print("Graph Loaded.  Loading Tensors")

        x = sess.graph.get_tensor_by_name("x:0")
        y_ = sess.graph.get_tensor_by_name("y_:0")
        keep_prob = tf.placeholder(tf.float32)

        keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
        predictator = sess.graph.get_tensor_by_name("predictator:0")

        print("Load complete!")
        print("End Time:   ", datetime.datetime.now())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check for basic prediction here
        file_contents = tf.read_file(image_name)
        read_image = tf.image.decode_jpeg(file_contents, channels=1)
        image = tf.image.resize_images(read_image, [200, 200])

        with sess.as_default():
            flatImage = image.eval().ravel()
        flatImage = np.multiply(flatImage, 1.0 / 255.0)
        self.predictImages[0] = flatImage
        self.predictLabels[0] = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        predict_it = sess.run(predictator, feed_dict={x: self.predictImages, y_: self.predictLabels, keep_prob: 1.0})

        prediction = self.labelMaster[predict_it]
        print("prediction: ", bytes.decode(prediction[0]))

        return bytes.decode(prediction[0])

predictor = ImagePredictor()
