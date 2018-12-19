
package Bags2DShapelets;

import static Bags2DShapelets.pngs2arff.loadPNGsInDirectory;
import development.Experiments;
import java.util.Arrays;
import weka.classifiers.meta.RotationForest;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class TestBed {

    public static void main(String[] args) throws Exception {
        firstTest();
    }
    
    public static void firstTest() throws Exception {
        
        
        pngs2arff.targetHeight = pngs2arff.targetWidth = 256;
        Instances data = pngs2arff.loadPNGsInDirectory("C:\\JamesLPHD\\BAGS\\psudo2Ddatabase\\", "C:\\JamesLPHD\\BAGS\\psudo2Ddatabase\\labels_threatOrNot.txt", Arrays.asList(new String[] {"threat","noThreat"}));
        
        int numcases = data.numInstances();
        double correct = .0;
        for (int i = 0; i < numcases; i++) {
            System.out.println("Fold " + i);
            
            Instances train = new Instances(data);

            Instances testContainer = new Instances(train,0);
            testContainer.add(train.remove(i));
            Instance test = testContainer.instance(0);

            
            System.out.println("Building");
            ST2D_Classifier c = new ST2D_Classifier();
            c.classifier = Experiments.setClassifier("RotF", i);
            c.setK(100);
            c.setNumShapeletsToSearch(1000);
            c.buildClassifier(train);

            System.out.println("Testing");
            double[] dist = c.distributionForInstance(test); 
            
            double act = test.classValue();
            double pred = utilities.GenericTools.indexOfMax(dist);
            if (pred == act)
                correct++;
            
            System.out.println(act + "," + pred + ",," + Arrays.toString(dist));
            
        }
        
        System.out.println("Accuracy: " + (correct / numcases));
    }
    
}
