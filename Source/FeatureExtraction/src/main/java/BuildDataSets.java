import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by AC010168 on 3/20/2017.
 */
public class BuildDataSets {

    public final static String[] CATEGORIES = {"dccomics", "doctorwho", "firefly", "lotr", "shelf",
            "spiderman", "starwars", "table", "weapon"};

    public final static int TRAIN_SAMPLE_COUNT = 50;  //1000
    public final static int TEST_SAMPLE_COUNT  = 5;

    public final static String SOURCE_PATH      = "C:/Users/ac010168/Desktop/train/";
    public final static String DATASET_PATH_ONE = "C:/Users/ac010168/Desktop/dataset4/";
    public final static String DATASET_PATH_TWO = "C:/Users/ac010168/Desktop/dataset5/";

    public static void main(String[] args) throws Exception {
        List<String> allImages = null;

        for (String category : CATEGORIES) {
            File sourceDir = new File(SOURCE_PATH + category);
            allImages = new ArrayList<String>(Arrays.asList(sourceDir.list()));

            System.out.println ("There are " + allImages.size() + " images for category " + category);
            System.out.println ("  " + allImages.get(0));

            List<String> ds1TrainList = new ArrayList<String>(TRAIN_SAMPLE_COUNT);
            List<String> ds2TrainList = new ArrayList<String>(TRAIN_SAMPLE_COUNT);

            List<String> ds1TestList  = new ArrayList<String>(TEST_SAMPLE_COUNT);
            List<String> ds2TestList  = new ArrayList<String>(TEST_SAMPLE_COUNT);

            Random rand = new Random();

            //Select Training Datasets First
            for (int i = 0; i < TRAIN_SAMPLE_COUNT; i++) {
                int curIndex = rand.nextInt(allImages.size());

                File curImage = new File(SOURCE_PATH + category + "/" + allImages.get(curIndex));
                File tgtImage = new File(DATASET_PATH_ONE + "train/" + category + "/" + allImages.get(curIndex));

                System.out.println ("Including in sample: " + allImages.get(curIndex));

                Path sourcePath = curImage.toPath();
                Path targetPath = tgtImage.toPath();
                Files.copy(sourcePath, targetPath, StandardCopyOption.REPLACE_EXISTING);

                allImages.remove(curIndex);
            }

            //Select Training Datasets First
            for (int i = 0; i < TRAIN_SAMPLE_COUNT; i++) {
                int curIndex = rand.nextInt(allImages.size());

                File curImage = new File(SOURCE_PATH + category + "/" + allImages.get(curIndex));
                File tgtImage = new File(DATASET_PATH_TWO + "train/" + category + "/" + allImages.get(curIndex));

                System.out.println ("Including in sample: " + allImages.get(curIndex));

                Path sourcePath = curImage.toPath();
                Path targetPath = tgtImage.toPath();
                Files.copy(sourcePath, targetPath, StandardCopyOption.REPLACE_EXISTING);

                allImages.remove(curIndex);
            }

            //Select Training Datasets First
            for (int i = 0; i < TEST_SAMPLE_COUNT; i++) {
                int curIndex = rand.nextInt(allImages.size());

                File curImage = new File(SOURCE_PATH + category + "/" + allImages.get(curIndex));
                File tgtImage = new File(DATASET_PATH_ONE + "test/" + category + "/" + allImages.get(curIndex));

                System.out.println ("Including in sample: " + allImages.get(curIndex));

                Path sourcePath = curImage.toPath();
                Path targetPath = tgtImage.toPath();
                Files.copy(sourcePath, targetPath, StandardCopyOption.REPLACE_EXISTING);

                allImages.remove(curIndex);
            }

            //Select Training Datasets First
            for (int i = 0; i < TEST_SAMPLE_COUNT; i++) {
                int curIndex = rand.nextInt(allImages.size());

                File curImage = new File(SOURCE_PATH + category + "/" + allImages.get(curIndex));
                File tgtImage = new File(DATASET_PATH_TWO + "test/" + category + "/" + allImages.get(curIndex));

                System.out.println ("Including in sample: " + allImages.get(curIndex));

                Path sourcePath = curImage.toPath();
                Path targetPath = tgtImage.toPath();
                Files.copy(sourcePath, targetPath, StandardCopyOption.REPLACE_EXISTING);

                allImages.remove(curIndex);
            }

        }

    }


}
