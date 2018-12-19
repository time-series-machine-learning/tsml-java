
package timeseriesweka.classifiers.ensembles.weightings;

import timeseriesweka.classifiers.ensembles.EnsembleModule;

/**
 * Will call findMeanAuroc() on each module's results, therefore not necessary to call
 * it within HESCA/whatever ensemble
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class AUROC extends ModuleWeightingScheme {

    private double power = 1.0;
    
    public AUROC() {
        uniformWeighting = true;
        needTrainPreds = true;
    }
    
    public AUROC(double power) {
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
        return makeUniformWeighting(Math.pow(module.trainResults.findMeanAUROC(), power), numClasses);
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName() + (power==1.0 ? "" : "(" + power + ")");
    }
    
}

