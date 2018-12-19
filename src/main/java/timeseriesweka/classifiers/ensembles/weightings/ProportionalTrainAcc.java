package timeseriesweka.classifiers.ensembles.weightings;

import timeseriesweka.classifiers.ensembles.EnsembleModule;

/**
 * Each modules weighting is defined as it's accuracy proportional to the other modules, 
 * i.e weight_i = (acc_i) / (sum of all accs)
 * 
 * @author James Large james.large@uea.ac.uk
 */
public class ProportionalTrainAcc extends ModuleWeightingScheme {

    protected double sumOfTrainAccs = 0.;
    
    public ProportionalTrainAcc() {
        uniformWeighting = true;
        needTrainPreds = false;
    }
    
    public void defineWeightings(EnsembleModule[] modules, int numClasses) {
        for (EnsembleModule m : modules) 
            sumOfTrainAccs += m.trainResults.acc;
        
        for (EnsembleModule m : modules) 
            m.posteriorWeights = defineWeighting(m, numClasses);
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        return makeUniformWeighting(module.trainResults.acc / sumOfTrainAccs, numClasses);
    }
    
}
