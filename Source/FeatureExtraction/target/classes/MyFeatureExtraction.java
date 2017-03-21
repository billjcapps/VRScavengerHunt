import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by AC010168 on 3/20/2017.
 */
public class MyFeatureExtraction {

    public final static String[] CATEGORIES = {"dccomics", "doctorwho", "firefly", "lotr", "shelf",
            "spiderman", "starwars", "table", "weapon"};

    public final static String TRAIN_ROOT  = "dataset1/train/";
    public final static String TEST_ROOT   = "dataset1/test/";
    public final static String OUTPUT_ROOT = "dataset1/output/";

    public static BufferedWriter trainWriter, testWriter;

    public static void main(String args[]) throws Exception {
        File trainOutput = new File(OUTPUT_ROOT + "trainData.txt");
        if (trainOutput.exists()) trainOutput.delete();
        File testOutput = new File(OUTPUT_ROOT + "testData.txt");
        if (testOutput.exists()) testOutput.delete();

        FileWriter writer1 = new FileWriter(trainOutput);
        FileWriter writer2 = new FileWriter(testOutput);

        trainWriter = new BufferedWriter(writer1);
        testWriter  = new BufferedWriter(writer2);

        DoGSIFTEngine doGSIFTEngine = new DoGSIFTEngine();

        int classifierCount = 0;
        for (String category : CATEGORIES) {
            System.out.println ("*******************************************************************");
            System.out.println ("Processing category: " + category);
            System.out.println ("*******************************************************************");
            int imageCount = 0;

            //Process all Training Images for this category
            File sourceDir = new File(TRAIN_ROOT + category);
            List<String> allImages = new ArrayList<String>(Arrays.asList(sourceDir.list()));

            for (String imageName : allImages) {
                System.out.println ("Processing " + category + " training image " + ++imageCount);

                MBFImage curImage = ImageUtilities.readMBF(new File(TRAIN_ROOT + category + "/" + imageName));

                LocalFeatureList<Keypoint> curFeatures = doGSIFTEngine.findFeatures(curImage.flatten());

                for (int i = 0; i < curFeatures.size(); i++) {
                    double c[] = curFeatures.get(i).getFeatureVector().asDoubleVector();
                    trainWriter.write(classifierCount + ",");
                    for (int j = 0; j < c.length; j++) {
                        trainWriter.write(c[j] + " ");
                    }
                    trainWriter.newLine();
                }
                trainWriter.flush();
            }

            //Now process all Test Images for this category
            File sourceDir2 = new File(TEST_ROOT + category);
            List<String> allImages2 = new ArrayList<String>(Arrays.asList(sourceDir2.list()));
            imageCount = 0;

            for (String imageName : allImages2) {
                System.out.println ("Processing " + category + " test image " + ++imageCount);

                MBFImage curImage = ImageUtilities.readMBF(new File(TEST_ROOT + category + "/" + imageName));

                LocalFeatureList<Keypoint> curFeatures = doGSIFTEngine.findFeatures(curImage.flatten());

                for (int i = 0; i < curFeatures.size(); i++) {
                    double c[] = curFeatures.get(i).getFeatureVector().asDoubleVector();
                    testWriter.write(classifierCount + ",");
                    for (int j = 0; j < c.length; j++) {
                        testWriter.write(c[j] + " ");
                    }
                    testWriter.newLine();
                }
                testWriter.flush();
            }
            classifierCount++;
        }

        trainWriter.flush();
        trainWriter.close();
        testWriter.flush();
        testWriter.close();
    }
}
