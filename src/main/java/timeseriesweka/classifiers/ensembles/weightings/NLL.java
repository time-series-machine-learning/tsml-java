
package timeseriesweka.classifiers.ensembles.weightings;

import timeseriesweka.classifiers.ensembles.EnsembleModule;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class NLL extends ModuleWeightingScheme {

    private double power = 1.0;
    
    public NLL() {
        uniformWeighting = true;
        needTrainPreds = true;
    }
    
    public NLL(double power) {
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
    
    public void defineWeightings(EnsembleModule[] modules, int numClasses) {
        double[] nlls = new double[modules.length];
        double sum = .0;
        
        for (int i = 0; i < nlls.length; i++) {
            nlls[i] = Math.pow(modules[i].trainResults.findNLL(), power);
            sum += nlls[i];
        }
        
        for (int i = 0; i < nlls.length; i++) {
            nlls[i] /= sum;
            nlls[i] = 1 - nlls[i];
            modules[i].posteriorWeights = makeUniformWeighting(nlls[i], numClasses);
        }
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        throw new UnsupportedOperationException("NLL weighting cannot be defined for a single module, "
                + "only in relation to the rest, call defineWeighings(...)");
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName() + (power==1.0 ? "" : "(" + power + ")");
    }
}