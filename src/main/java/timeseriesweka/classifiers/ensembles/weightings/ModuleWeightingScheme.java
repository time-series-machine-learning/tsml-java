package timeseriesweka.classifiers.ensembles.weightings;

import timeseriesweka.classifiers.ensembles.EnsembleModule;

/**
 * Base class for defining the weighting of a classifiers votes in ensemble classifiers
 * 
 * @author James Large
 */
public abstract class ModuleWeightingScheme {
    
    public boolean uniformWeighting = true;
    public boolean needTrainPreds = true;
    
    public void defineWeightings(EnsembleModule[] modules, int numClasses) {
        for (EnsembleModule m : modules) //by default, sets weights independently for each module
            m.posteriorWeights = defineWeighting(m, numClasses);
        
        //some schemes may sets weights for each moduel relative to the rest, and 
        //so will need to override this method
    }
    
    protected abstract double[] defineWeighting(EnsembleModule trainPredictions, int numClasses);
    
    protected double[] makeUniformWeighting(double weight, int numClasses) {
        double[] weights = new double[numClasses];
        for (int i = 0; i < weights.length; ++i)
            weights[i] = weight;
        return weights;
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName();
    }
    
}
