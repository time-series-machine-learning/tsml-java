package timeseriesweka.classifiers.ensembles.voting.stacking;

import weka.classifiers.Classifier;
import timeseriesweka.classifiers.ensembles.EnsembleModule;
import weka.core.DenseInstance;
import weka.core.Instance;

/**
 *
 * @author James Large james.large@uea.ac.uk
 */
public class StackingOnDists extends AbstractStacking {

    public StackingOnDists(Classifier classifier) {
        super(classifier);
    }
    
    public StackingOnDists(Classifier classifier, int numClasses) {
        super(classifier, numClasses);
    }
    
    @Override
    protected void setNumOutputAttributes(EnsembleModule[] modules) {
        this.numOutputAtts = modules.length*numClasses + 1; //each dist + class val
    }
    
    @Override
    protected Instance buildInst(double[][] dists, Double classVal) {
        double[] instData = new double[numOutputAtts];
        
        int i = 0;
        for (int m = 0; m < dists.length; m++) 
            for (int c = 0; c < numClasses; c++) 
                instData[i++] = dists[m][c];
        
        assert(i == numOutputAtts-2);
        
        if (classVal != null)
            instData[numOutputAtts-1] = classVal; 
        //else irrelevent 
        
        instsHeader.add(new DenseInstance(1.0, instData));
        return instsHeader.remove(0);
    }
    
}
