package timeseriesweka.classifiers.ensembles.voting.stacking;

import weka.classifiers.Classifier;
import timeseriesweka.classifiers.ensembles.EnsembleModule;
import weka.core.DenseInstance;
import weka.core.Instance;

/**
 *
 * @author James Large james.large@uea.ac.uk
 */
public class StackingOnPreds extends AbstractStacking {
    
    public StackingOnPreds(Classifier classifier) {
        super(classifier);
    }
    
    public StackingOnPreds(Classifier classifier, int numClasses) {
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
            instData[m] = indexOfMax(dists[m]);
        
        if (classVal != null)
            instData[numOutputAtts-1] = classVal; 
        //else irrelevent 
        
        instsHeader.add(new DenseInstance(1.0, instData));
        return instsHeader.remove(0);
    }

}
