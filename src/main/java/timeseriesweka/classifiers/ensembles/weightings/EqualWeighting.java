package timeseriesweka.classifiers.ensembles.weightings;

import timeseriesweka.classifiers.ensembles.EnsembleModule;


/**
 *
 * Gives equal weights to all modules, i.e simple majority vote
 * 
 * @author James Large
 */
public class EqualWeighting extends ModuleWeightingScheme {

    public EqualWeighting() {
        uniformWeighting = true;
        needTrainPreds = false;
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule trainPredictions, int numClasses) {
        return makeUniformWeighting(1.0, numClasses);
    }
    
}
