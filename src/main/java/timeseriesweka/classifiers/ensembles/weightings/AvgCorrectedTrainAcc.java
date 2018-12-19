
package timeseriesweka.classifiers.ensembles.weightings;

import timeseriesweka.classifiers.ensembles.EnsembleModule;

/**
 * Sets each module's weighting to Math.max(0.01, module.trainResults.acc - maxClassWeighting),
 * where maxClassWeighting is the proportion of cases belonging to the most common class, 
 * i.e the accuracy expected from a completely biased classifier
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class AvgCorrectedTrainAcc extends ModuleWeightingScheme {

    public AvgCorrectedTrainAcc() {
        uniformWeighting = true;
        needTrainPreds = false;
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        //made non zero (effectively 1% accuracy) in weird case that all classifiers get less than expected acc
        return makeUniformWeighting(Math.max(0.01, module.trainResults.acc - (1.0/numClasses)), numClasses);
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName();
    }
}
