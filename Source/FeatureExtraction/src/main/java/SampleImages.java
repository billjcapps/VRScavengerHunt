import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;

import java.io.File;

/**
 * Created by AC010168 on 3/20/2017.
 */
public class SampleImages {

    public final static int SAMPLE_WIDTH  = 400;
    public final static int SAMPLE_HEIGHT = 400;

    public final static int XOFFSET       = 3720;
    public final static int YOFFSET       = 310;

    public final static int XRANGE        = 105;
    public final static int YRANGE        = 230;

    public final static int XSAMPLE_RATE  = 1;
    public final static int YSAMPLE_RATE  = 2;

    public final static String CATEGORY   = "spiderman";

    public final static String OUTPUT_DIR = "C:\\Users\\ac010168\\Desktop\\train\\" + CATEGORY + "\\";

    public static void main(String[] args) {
        try {
            MBFImage masterImage = ImageUtilities.readMBF(new File("C:\\Users\\ac010168\\Desktop\\vr images\\IMG_20170215_223634.vr.jpg"));

            MBFImage sampleImage = new MBFImage(400, 400);
            int imageTag = 0;

            int totalImages = (XRANGE / XSAMPLE_RATE) * (YRANGE / YSAMPLE_RATE);
            for (int i = 0; i < XRANGE; i += XSAMPLE_RATE) {
                for (int j = 0; j < YRANGE; j += YSAMPLE_RATE) {
                    int xOffset = XOFFSET + i;
                    int yOffset = YOFFSET + j;

                    for (int x = 0; x < SAMPLE_WIDTH; x++ ) {
                        for (int y = 0; y < SAMPLE_HEIGHT; y++) {
                            sampleImage.setPixel(x, y, masterImage.getPixel(x + xOffset, y + yOffset));
                        }
                    }
                    imageTag++;

                    String imageName = "image_";
                    if (imageTag < 10000) imageName += "0";
                    if (imageTag < 1000)  imageName += "0";
                    if (imageTag < 100)   imageName += "0";
                    if (imageTag < 10)    imageName += "0";
                    imageName += "" + imageTag + ".jpg";

                    System.out.println ("Printing " + CATEGORY + " " + imageName + "  (out of " + totalImages + ")");
                    ImageUtilities.write(sampleImage, new File(OUTPUT_DIR + imageName));
                }
            }
        } catch(Throwable t) {
            t.printStackTrace();
        }
    }
}
