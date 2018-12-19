package timeseriesweka.classifiers.ensembles.weightings;

import timeseriesweka.classifiers.ensembles.EnsembleModule;

/**
 * Will call findBalancedAcc() on each module's results, therefore not necessary to call
 * it within HESCA/whatever ensemble
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class BalancedAccuracy extends ModuleWeightingScheme {

    private double power = 1.0;
    
    public BalancedAccuracy() {
        uniformWeighting = true;
        needTrainPreds = true;
    }
    
    public BalancedAccuracy(double power) {
        this.power = power;
        uniformWeighting = true;
        needTrainPreds = true;
    }
    
    public double getPower() { 
        return power;
    }
    
    public void setPower(double power) {
        this.power = power;
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        module.trainResults.findAllStats(); //countsPerClass not initialised without this call
        return makeUniformWeighting(Math.pow(module.trainResults.balancedAcc, power), numClasses);
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName() + (power==1.0 ? "" : "(" + power + ")");
    }
    
}
