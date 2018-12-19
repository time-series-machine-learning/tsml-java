
package Bags2DShapelets;

import timeseriesweka.classifiers.ensembles.elastic_ensemble.ED1NN;
import vector_classifiers.CAWPE;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class ST2D_Classifier extends AbstractClassifier {

    ST2D transform = null;
    Classifier classifier = null;
    
    int seed = 0;
    
    Instances trainData = null;
    Instances testFormat = null;
    
    
    public ST2D_Classifier() { 
        transform = new ST2D();
        //classifier = new CAWPE();
        classifier = new ED1NN();
    }
    
    public void setNumShapeletsToSearch(int numShapeletsToSearch) {
        transform.numShapeletsToSearch = numShapeletsToSearch;
    }
    public int getNumShapeletsToSearch() {
        return transform.numShapeletsToSearch;
    }
    
    public void setK(int k) {
        transform.k = k;
    }
    public int getK() {
        return transform.k;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainData = new Instances(data);
        testFormat = new Instances(trainData, 0);
        
        Instances transformedData = transform.process(trainData);        
        classifier.buildClassifier(transformedData);
    }
    
    @Override
    public double[] distributionForInstance(Instance inst) throws Exception {
        testFormat.add(inst);
        Instances transformed = transform.process(testFormat);
        testFormat.remove(0);
        
        return classifier.distributionForInstance(transformed.get(0));
    }

}
