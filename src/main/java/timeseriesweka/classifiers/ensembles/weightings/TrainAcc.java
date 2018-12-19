
package timeseriesweka.classifiers.ensembles.weightings;

import timeseriesweka.classifiers.ensembles.EnsembleModule;

/**
 * Simply uses the modules train acc as it's weighting. Extension: can raise the accuracy
 * to some power in order to scale the relative difference in accuracy between modules 
 * 
 * e.g, if raising all accuracies to power of 2, weights (0.7, 0.8, 0.9) become (0.49, 0.64, 0.81)
 * 
 * @author James Large
 */
public class TrainAcc extends ModuleWeightingScheme {

    private double power = 1.0;
    
    public TrainAcc() {
        uniformWeighting = true;
        needTrainPreds = false;
    }
    
    public TrainAcc(double power) {
        this.power = power;
        uniformWeighting = true;
        needTrainPreds = false;
    }
    
    public double getPower() { 
        return power;
    }
    
    public void setPower(double power) {
        this.power = power;
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        return makeUniformWeighting(Math.pow(module.trainResults.acc, power), numClasses);
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName() + (power==1.0 ? "" : "(" + power + ")");
    }
    
}
