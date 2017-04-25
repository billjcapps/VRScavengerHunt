import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.bytedeco.javacpp.indexer.{DoubleIndexer, FloatRawIndexer, UByteRawIndexer}
import org.bytedeco.javacpp.opencv_core.KeyPointVector
import org.bytedeco.javacpp.opencv_features2d.{AKAZE, BFMatcher, BOWImgDescriptorExtractor, FlannBasedMatcher}
import org.bytedeco.javacpp.opencv_core.{CV_32F, Mat}
import org.bytedeco.javacpp.opencv_imgcodecs._
import org.bytedeco.javacpp.opencv_xfeatures2d.SIFT

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * Created by pradyumnad on 17/07/15.
  *
  * Uplifted to newest JavaCPP / OpenCV libraries by Adam Carter on 2/24/17
  */
object ImageUtils {

  var count = 0
  var count2 = 0

  def descriptors(file: String): Mat = {
    count = count + 1
    println ("Image Name [" + count + "]: " + file)

    val img_1 = imread(file, CV_LOAD_IMAGE_GRAYSCALE)
    if (img_1.empty()) {
      println("Image is empty")
      -1
    }
    //-- Step 1: Detect the keypoints using ORB
    val detector = AKAZE.create()
    val keypoints_1 = new KeyPointVector()

    val mask = new Mat
    val descriptors = new Mat

    //detector.detectAndCompute(img_2, mask, keypoints_1, descriptors)
    detector.detectAndCompute(img_1, mask, keypoints_1, descriptors)
    descriptors.convertTo(descriptors, CV_32F);

    //    println(s"No of Keypoints ${keypoints_1.size()}")
    //println(s"Key Descriptors ${descriptors.rows()} x ${descriptors.cols()}")
    descriptors
  }

  def bowDescriptors(file: String, dictionary: Mat): Mat = {
    count2 = count2 + 1

    println ("Generating Histogram for image [" + count2 + "]:" + file)
    try {
      val matcher = new BFMatcher()
      val detector = AKAZE.create()
      val extractor = AKAZE.create()
      val bowDE = new BOWImgDescriptorExtractor(extractor, matcher)
      bowDE.setVocabulary(dictionary)
      println(bowDE.descriptorSize() + " " + bowDE.descriptorType())

      val img = imread(file, CV_LOAD_IMAGE_GRAYSCALE)
      if (img.empty()) {
        println("Image is empty")
        -1
      }

      val keypoints = new KeyPointVector()
      val mask = new Mat
      val descriptors = new Mat

      detector.detectAndCompute(img, mask, keypoints, descriptors)
      descriptors.convertTo(descriptors, CV_32F);

      val response_histogram = new Mat
      bowDE.compute(descriptors, response_histogram)

      //println("Histogram : " + response_histogram.asCvMat().toString)
      //println("Histogram : " + response_histogram.toString)
      response_histogram
    } catch {
      case _:Error => null
    }

  }

  def matToVector(mat: Mat): Vector = {
    //TODO - Need to figure out how to grab single pixels like CvMat is
    //val imageCvmat = mat.asCvMat()
    //val noOfCols = imageCvmat.cols()
    //http://stackoverflow.com/questions/33035781/access-to-the-pixel-value-of-a-mat-using-the-javacv-api
    val indexer = mat.createIndexer().asInstanceOf[FloatRawIndexer]
    val noOfCols = indexer.cols().toInt

    //Channels discarded, take care of this when you are using multiple channels
    val imageInDouble = new Array[Double](noOfCols)
    for (col <- 0 to noOfCols - 1) {
      //val pixel = imageCvmat.get(0, col)
      val pixel = indexer.get(col.toLong)
      imageInDouble(col) = pixel
    }
    val featureVector = new DenseVector(imageInDouble)
    featureVector
  }

  def matToVectors(mat: Mat): Array[Vector] = {
    //val imageCvmat = mat.asCvMat()
    val indexer = mat.createIndexer().asInstanceOf[FloatRawIndexer]
    val noOfCols = indexer.cols().toInt
    val noOfRows = indexer.rows().toInt

    val fVectors = new ArrayBuffer[DenseVector]()
    //Channels discarded, take care of this when you are using multiple channels

    for (row <- 0 to noOfRows - 1) {
      val imageInDouble = new Array[Double](noOfCols.toInt)
      for (col <- 0 to noOfCols - 1) {
        //val pixel = imageCvmat.get(row, col)
        val pixel = indexer.get(row.toLong, col.toLong)
        imageInDouble :+ pixel
      }
      val featureVector = new DenseVector(imageInDouble)
      fVectors :+ featureVector
    }

    fVectors.toArray
  }

  def matToDoubles(mat: Mat): Array[Array[Double]] = {
    //val imageCvmat = mat.asCvMat()
    val indexer = mat.createIndexer().asInstanceOf[FloatRawIndexer]
    val noOfCols = indexer.cols().toInt
    val noOfRows = indexer.rows().toInt

    val fVectors = new ArrayBuffer[Array[Double]]()
    //Channels discarded, take care of this when you are using multiple channels

    for (row <- 0 to noOfRows - 1) {
      val imageInDouble = new Array[Double](noOfCols.toInt)
      for (col <- 0 to noOfCols - 1) {
        //val pixel = imageCvmat.get(row, col)
        val pixel = indexer.get(row.toLong, col.toLong)
        imageInDouble :+ pixel
      }
      fVectors :+ imageInDouble
    }
    fVectors.toArray
  }

  def matToString(mat: Mat): List[String] = {
    //val imageCvmat = mat.asCvMat()
    val indexer = mat.createIndexer().asInstanceOf[FloatRawIndexer]
    val noOfCols = indexer.cols().toInt
    val noOfRows = indexer.rows().toInt

    val fVectors = new mutable.MutableList[String]
    //Channels discarded, take care of this when you are using multiple channels

    for (row <- 0 to noOfRows - 1) {
      val vecLine = new StringBuffer("")
      for (col <- 0 to noOfCols - 1) {
        //val pixel = imageCvmat.get(row, col)
        val pixel = indexer.get(row.toLong, col.toLong)
        vecLine.append(pixel + " ")
      }

      fVectors += vecLine.toString
    }
    fVectors.toList
  }

  def vectorsToMat(centers: Array[Vector]): Mat = {

    val vocab = new Mat(centers.size, centers(0).size, CV_32F)
    val indexer = vocab.createIndexer().asInstanceOf[FloatRawIndexer]

    var i = 0
    for (c <- centers) {

      var j = 0
      for (v <- c.toArray) {
        //vocab.asCvMat().put(i, j, v)
        indexer.put(i.toLong, j.toLong, v.toFloat)
        j += 1
      }
      i += 1
    }

    //    println("The Mat is")
    //    println(vocab.asCvMat().toString)

    vocab
  }

}
