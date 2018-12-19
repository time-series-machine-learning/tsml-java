
package Bags2DShapelets;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import javax.imageio.ImageIO;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class pngs2arff {
    
    public static int targetWidth = 50;
    public static int targetHeight = 50;
    
    public static Instances loadPNGsInDirectory(String directoryPath, String classLabelFilePath, List<String> classStrings) throws Exception {
        
        File[] imgPaths = (new File(directoryPath)).listFiles((pathname) -> {
            return pathname.isFile() && pathname.getName().endsWith(".png");
        });
        
        double[] classVals = new double[imgPaths.length];
        Scanner in = new Scanner(new File(classLabelFilePath));
        int it = 0;
        while (in.hasNextLine())
            classVals[it++] = Double.parseDouble(in.nextLine());
        assert(it==classVals.length);
        
        ArrayList<Attribute> atts = new ArrayList<>(targetHeight*targetWidth + 1);
        for (int y = 0; y < targetHeight; y++)
            for (int x = 0; x < targetWidth; x++)
                atts.add(new Attribute("px" + ((y*targetHeight)+x) + ":" + x + "_" + y));
        atts.add(new Attribute("class", classStrings));
        
        Instances unidata = new Instances("Images", atts, imgPaths.length);
        unidata.setClassIndex(unidata.numAttributes()-1);
        
        it = 0;
        for (File imgPath : imgPaths)
            unidata.add(new DenseInstance(1.0, loadFlattenedPNG(imgPath.getAbsolutePath(), classVals[it++])));
        
        Instances multidata = utilities.multivariate_tools.MultivariateInstanceTools.convertUnivariateToMultivariate(unidata, targetWidth);
        
        return multidata;
    }
    
    /**
     * Builds a 1d double array that is in the format that would be expected by 
     * utilities.multivariate_tools.MultivariateInstanceTools.convertUnivariateToMultivariate
     * that is, each row in a single block one by one, with AN EXTRA SPACE FOR THE CLASS VALUE ONCE AT THE END
     * the class value will default to -1.0
     * 
     * 1 1 1 
     * 2 2 2   =>    1 1 1 2 2 2 3 3 3 label
     * 3 3 3 
     * 
     */
    public static double[] loadFlattenedPNG(String filePath) throws Exception {
        return loadFlattenedPNG(filePath, -1.0);
    }
    
    /**
     * Builds a 1d double array that is in the format that would be expected by 
     * utilities.multivariate_tools.MultivariateInstanceTools.convertUnivariateToMultivariate
     * that is, each row in a single block one by one, with the class value once at the end
     * 1 1 1 
     * 2 2 2   =>    1 1 1 2 2 2 3 3 3 label
     * 3 3 3 
     * 
     */
    public static double[] loadFlattenedPNG(String filePath, double label) throws Exception {
        if (!filePath.endsWith(".png"))
            filePath += ".png";
        
        BufferedImage img = null;
        try {
            img = ImageIO.read(new File(filePath));
            img = resize(img, targetWidth, targetHeight);
        } 
        catch (IOException e) {
            System.out.println("Couldn't load image file: " + filePath);
            throw e;
        }
        
        int width = img.getWidth();
        int height = img.getHeight();
        
        double[] rawFlattenedImgData = new double[width*height + 1]; //+1 for label
        rawFlattenedImgData[rawFlattenedImgData.length-1] = label;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // / 255 will just put in range 0 to 1
                rawFlattenedImgData[(y*height)+x] = img.getRGB(x,y) / 255.0;
            }
        }
        
        double[] result = normalise(rawFlattenedImgData);
        
        return result;
    }
    
    //https://memorynotfound.com/java-resize-image-fixed-width-height-example/
    private static BufferedImage resize(BufferedImage img, int height, int width) {
        Image tmp = img.getScaledInstance(width, height, Image.SCALE_SMOOTH);
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = resized.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();
        return resized;
    }
    
    /**
     * normalises into range [0..1]
     */
    private static double[] normalise(double[] in) {
        double[] out = new double[in.length];
        out[out.length-1] = in[in.length-1]; //label
        
        double min = utilities.GenericTools.min(in);
        double max = utilities.GenericTools.max(in);
        
        for (int i = 0; i < out.length-1; i++)
            out[i] = (in[i] - min) / (max - min); 
        
        return out;
    }
    
    public static void main(String[] args) throws Exception {

        Instances multiVariateImageInstances = loadPNGsInDirectory("C:\\JamesLPHD\\BAGS\\psudo2Ddatabase\\", "C:\\JamesLPHD\\BAGS\\psudo2Ddatabase\\labels_threatOrNot.txt", Arrays.asList(new String[] {"threat","noThreat"}));
        
        System.out.println(multiVariateImageInstances.numInstances());
        System.out.println(multiVariateImageInstances.numAttributes());
        System.out.println(multiVariateImageInstances.numClasses());
        System.out.println(multiVariateImageInstances.get(0));
    }
}
