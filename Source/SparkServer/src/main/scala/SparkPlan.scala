import java.io.{ByteArrayInputStream, File}
import javax.imageio.ImageIO

import unfiltered.filter.Plan
import _root_.unfiltered.request.{Body, GET, POST, Path}
import sun.misc.BASE64Decoder
import unfiltered.response.{Ok, ResponseString}

/**
  * Created by AC010168 on 2/20/2017.
  */
object SparkPlan extends Plan {

  def intent = {
    case req@GET(Path("/spark/test")) => {
      val testImage1 = "dccomics: " + ServerSupport.testImage("data/test/dccomics/image_02266.jpg")
      val testImage2 = "doctorwho: " + ServerSupport.testImage("data/test/doctorwho/image_04252.jpg")
      val testImage3 = "firefly: " + ServerSupport.testImage("data/test/firefly/image_00924.jpg")
      val testImage4 = "lotr: " + ServerSupport.testImage("data/test/lotr/image_00915.jpg")
      val testImage5 = "shelf: " + ServerSupport.testImage("data/test/shelf/image_10163.jpg")
      val testImage6 = "spiderman: " + ServerSupport.testImage("data/test/spiderman/image_05367.jpg")
      val testImage7 = "starwars: " + ServerSupport.testImage("data/test/starwars/image_08814.jpg")
      val testImage8 = "table: " + ServerSupport.testImage("data/test/table/image_01492.jpg")
      val testImage9 = "weapon: " + ServerSupport.testImage("data/test/weapon/image_08249.jpg")

      val result = testImage1 + "\n" + testImage2 + "\n" + testImage3 + "\n" +
        testImage4 + "\n" + testImage5 + "\n" + testImage6 + "\n" +
        testImage7 + "\n" + testImage8 + "\n" + testImage9
        Ok ~> ResponseString(result)
    }

    case req@POST(Path("/spark/predict")) => {
      try {
        val imageByte = (new BASE64Decoder()).decodeBuffer(Body.string(req));
        val bytes = new ByteArrayInputStream(imageByte)
        val image = ImageIO.read(bytes);
        ImageIO.write(image, "png", new File("data/web/image.png"))
      } catch {
        case e : Exception => println (e.getMessage());
          e.printStackTrace();
      }
      Ok ~> ResponseString(ServerSupport.testImage("data/web/image.png"))
    }
  }
}
