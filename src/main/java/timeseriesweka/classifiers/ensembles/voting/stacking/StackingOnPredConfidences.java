package timeseriesweka.classifiers.ensembles.voting.stacking;

import weka.classifiers.Classifier;
import timeseriesweka.classifiers.ensembles.EnsembleModule;
import weka.core.DenseInstance;
import weka.core.Instance;

/**
 *  This is dumb/useless in current format, just makes a vector of confidences but don't know what class
 * those confidences refer too. Maybe revisit at some point and make the vector a list of pairs, (classval, confidence)
 * 
 * @author James Large james.large@uea.ac.uk
 */
public class StackingOnPredConfidences extends AbstractStacking {
    
    public StackingOnPredConfidences(Classifier classifier) {
        super(classifier);
    }
    
    public StackingOnPredConfidences(Classifier classifier, int numClasses) {
        super(classifier, numClasses);
    }
    
    @Override
    protected void setNumOutputAttributes(EnsembleModule[] modules) {
        this.numOutputAtts = modules.length + 1; //each pred + class val
    }
    
    @Override
    protected Instance buildInst(double[][] dists, Double classVal) throws Exception {
        double[] instData = new double[numOutputAtts];
        
        for (int m = 0; m < dists.length; m++) 
            instData[m] = dists[m][(int)indexOfMax(dists[m])];
        
        if (classVal != null)
            instData[numOutputAtts-1] = classVal; 
        //else irrelevent 
        
        instsHeader.add(new DenseInstance(1.0, instData));
        return instsHeader.remove(0);
    }
    
}
