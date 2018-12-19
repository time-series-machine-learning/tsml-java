
package timeseriesweka.classifiers.ensembles.weightings;

import timeseriesweka.classifiers.ensembles.EnsembleModule;

/**
 * Will define a module's weighting as it's train accuracy by default, however in 
 * cases where the class distribution of the dataset is 'uneven', it will instead use MCC 
 * 
 * Currently, 'uneven' is (arbitrarily) defined as one class having four times as many train instances
 * as another, e.g in the 2 class case one class having 80% of the train insts would lead to 
 * the MCC weighting being used.
 * 
 * @author James Large james.large@uea.ac.uk
 */
public class TrainAccOrMCC extends ModuleWeightingScheme {
    
    protected double unevenProp = 4.0; //how much bigger must max class dist be than min to use MCC 
    
    public TrainAccOrMCC() {
        uniformWeighting = true;
    }
    
    /**
     * @param minPropToUseMCC how much bigger must max class dist be than the min dist to use MCC, defaults to 4, 
     * i.e in a two class class MCC will be used if one class has 80% of the insts, and the other has 20%
     */
    public TrainAccOrMCC(double minPropToUseMCC) {
        uniformWeighting = true;
        this.unevenProp = minPropToUseMCC;
    }
    
    private ModuleWeightingScheme scheme = null;
    
    @Override
    public void defineWeightings(EnsembleModule[] modules, int numClasses) {            
        double[] dist = classDistribution(modules[0].trainResults.getTrueClassVals(), numClasses);
        
        double max = dist[0], min = dist[0];
        for (int c = 1; c < dist.length; c++) {
            if (dist[c] > max)
                max = dist[c];
            else if (dist[c] < min)
                min = dist[c];
        }
        
        if (max >= min*this.unevenProp)
            scheme = new MCCWeighting();
        else 
            scheme = new TrainAcc();
        
        for (EnsembleModule module : modules)
            module.posteriorWeights = defineWeighting(module, numClasses);
    }
    
    @Override
    protected double[] defineWeighting(EnsembleModule trainPredictions, int numClasses) {
        return scheme.defineWeighting(trainPredictions, numClasses);
    }
    
    protected double[] classDistribution(double[] classVals, int numClasses) {
        double[] dist = new double[numClasses];
        
        for (double c : classVals)
            ++dist[(int)c];
        
        for (int i = 0; i < numClasses; i++)
            dist[i] /= classVals.length;
            
        return dist;
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + "(" + unevenProp + ")";
    }
    
}
